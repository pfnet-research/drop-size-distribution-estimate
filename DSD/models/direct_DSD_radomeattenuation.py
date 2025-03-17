import os
from typing import List  # NOQA

import time
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
#import torchviz

from DSD.models.MPPAWR_DSD import network_init_MPPAWR_DSD_range_combined_6dim
#from data_assimilation.utils import compute_K, compute_K_sigma_block_diag
from einops import rearrange

import time
from torch import profiler  # NOQA

torch.set_default_dtype(torch.float32)

def compute_loss_integral5(
    h_results,
    x_value,
    sigma_err,
    mask,
):
    (N_data, N_sample, n_step, x_dim) = h_results.shape
    assert len(sigma_err) == x_dim
    #h_results_ = torch.stack([1*pow(10, h_results[..., 0]/10), h_results[..., 1], h_results[..., 2]], dim=-1) # dB scale
    #x_value_ = torch.stack([1*pow(10, x_value[..., 0]/10), x_value[..., 1], x_value[..., 2]], dim=-1) # dB scale
    h_results_ = h_results
    x_value_ = x_value
    diff = h_results_-x_value_.unsqueeze(1).repeat(1, N_sample, 1, 1)
    diff = torch.clip(diff, -50, 50) # cut phiDP difference at pm20
    gaussian_dist = dist.Normal(torch.zeros_like(h_results), sigma_err.view(1, 1, 1, -1))
    to_be_summed = gaussian_dist.log_prob(diff)*mask.unsqueeze(1).repeat(1, N_sample, 1, 1)
    loss0 = torch.sum(to_be_summed[..., 0])
    #loss1 = torch.sum(to_be_summed[..., 1])
    channel1_x_value = x_value[..., 1].unsqueeze(1).repeat(1, N_sample, 1, 1)
    loss1 = torch.sum(
        torch.where(
            (channel1_x_value == 0.3) & (diff[..., 1] < 0),
            torch.tensor(0.0, device=diff.device),
            to_be_summed[..., 1]
        )
    )
    loss2 = torch.sum(to_be_summed[..., 2])
    log_prob_values3 = loss0 + loss1 + loss2
    return log_prob_values3 / N_sample / N_data, loss0.detach() / N_sample / N_data, loss1.detach() / N_sample / N_data, loss2.detach() / N_sample / N_data

def compute_KL_sampling(mu_1, sigma_1, sigma_2, dynamics, z_dim, N_data, n_step, dynamics_args={}):
    # mu_1, sigma_1 should represent q(ht|o1:T): sampling from here
    # sigma_2 should represent p(ht|ht-1)
    (N_data, n_step, z_dim) = mu_1.shape
    assert mu_1.shape == (
        N_data,
        n_step,
        z_dim,
    ), f"mu_1.shape should be (N_data, n_step, z_dim) but is {mu_1.shape}"
    assert sigma_1.shape == (
        N_data,
        n_step,
        z_dim,
        z_dim,
    ), f"sigma_1.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_1.shape}"
    assert sigma_2.shape == (
        z_dim,
        z_dim,
    ), f"sigma_2.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_2.shape}"
    sqrt_sigma_1 = torch.linalg.cholesky(sigma_1)
    mvn1 = dist.Normal(torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda"))
    N_sample = 1
    points = mvn1.sample((N_data, n_step, N_sample, z_dim))
    result_ = mu_1.unsqueeze(2).repeat(1, 1, N_sample, 1) + torch.einsum(
        "abij,absj->absi", sqrt_sigma_1, points
    )  # (N_data, n_step, N_sample, z_dim), samples from q(ht|o1:T) and will work as the means of p(ht|ht-1)
    result_sliced = result_[:, :-1, :, :]
    zeros = torch.zeros(result_.shape[0], 1, result_.shape[2], result_.shape[3], device="cuda")
    result_ = torch.concat([zeros, result_sliced], dim=1) # samples from q(ht-1|o1:T), the first step is just zero
    result = dynamics(result_, **dynamics_args)
    #print(f"{result[:, :, 0, 0]=}")
    #print(f"{result[:, :, 0, 1]=}")
    #print(f"{result[:, :, 0, 2]=}")
    dist_a = torch.distributions.MultivariateNormal(mu_1.unsqueeze(2).repeat(1, 1, N_sample, 1), sigma_1.unsqueeze(2).repeat(1, 1, N_sample, 1, 1)) # q(ht|o1:T), sample shape:N_data, n_step, N_sample, z_dim
    dist_b = torch.distributions.MultivariateNormal(result, sigma_2.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(result.shape[0], result.shape[1], result.shape[2], 1, 1)) # p(ht|ht-1), ht-1 sampled from q(ht|o1:T)
    #print(f"{dist_a=}")
    #print(f"{dist_b=}")
    KLdivs = torch.distributions.kl.kl_divergence(dist_a, dist_b) # N_data, n_step, N_sample
    KLdivs_mean = torch.mean(KLdivs, dim=2) # N_data, n_step
    KLdivs_mean2 = torch.mean(KLdivs_mean, dim=0) # n_step
    loss = torch.sum(KLdivs_mean2)
    return loss


def compute_KL_gaussians(mu_1, sigma_1, mu_2, sigma_2, z_dim, N_data, n_step, sigma_block_diag=True):
    assert mu_1.shape == (
        N_data,
        n_step,
        z_dim,
    ), f"mu_1.shape should be (N_data, n_step, z_dim) but is {mu_1.shape}"
    assert sigma_1.shape == (
        N_data,
        n_step,
        z_dim,
        z_dim,
    ), f"sigma_1.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_1.shape}"
    assert mu_2.shape == (
        N_data,
        n_step,
        z_dim,
    ), f"mu_2.shape should be (N_data, n_step, z_dim) but is {mu_2.shape}"
    assert sigma_2.shape == (
        N_data,
        n_step,
        z_dim,
        z_dim,
    ), f"sigma_2.shape should be (N_data, n_step, z_dim, z_dim) but is {sigma_2.shape}"
    sigma_2inv = torch.inverse(sigma_2)
    loss1 = torch.einsum(
        "ijkk->ij", torch.einsum("ijkl,ijlm->ijkm", sigma_2inv, sigma_1)
    ).sum()
    loss2 = torch.einsum(
        "lik,lik->l",
        mu_2 - mu_1,
        torch.einsum("lij,lijk->lik", mu_2 - mu_1, sigma_2inv),
    ).sum()
    S1 = torch.linalg.svdvals(sigma_1)
    S2 = torch.linalg.svdvals(sigma_2)
    log_det_sigma_1 = torch.sum(torch.log(S1), 2)
    log_det_sigma_2 = torch.sum(torch.log(S2), 2)
    loss3 = (log_det_sigma_2 - log_det_sigma_1).sum() - z_dim * N_data * n_step
        
    assert ~torch.isnan(loss1), loss1
    assert ~torch.isnan(loss2), loss2
    assert ~torch.isnan(loss3), torch.min(torch.det(sigma_1))
    loss_value2 = 0.5 * (loss1 + loss2 + loss3)
    # assert loss_value2 >= 0, loss_value2
    loss_value2 = torch.max(torch.tensor(0.0), loss_value2)
    assert loss_value2 < 1e40, loss_value2
    return loss_value2 / N_data


class Direct_DDP(nn.Module):
    def __init__(
        self,
        mode,
        F,
        h_network,
        take_blockstep,
        take_loss_physical,
        log_sysnoise,
        log_obsnoise,
        G_val,
        z_dim,
        x_dim,
        time_series_input,
        time_series_length,
        aux_alpha,
        save_folder,
        architecture,
        periodic=False,
        periodic_indices=[],
        period=2*torch.pi,
        log_concentration_periodic=None,
        variable_kernelsize=False,
        unitmatrix=False,
        sigma_block_diag=True,
        G_nondiag=False,
        kernel=3,
        CZV=None,
        CkDP=None,
        ClAH=None,
        ClAV=None,
        lambda_AH=None,
        lambda_AV=None,
        PhiDP_offset_short=None,
        PhiDP_offset_long=None,
        h_network_fixed_params={},
        obs_mode="range_independent",
        f_nonlinear=False,
        f_nonlinear_tangents=None,
        Dynamics_allsteps=None,
    ):
        super().__init__()
        self.mode = mode
        self.f_nonlinear = f_nonlinear
        self.F = F
        self.h_network = h_network
        self.take_blockstep = take_blockstep
        self.take_loss_physical = take_loss_physical
        '''
        self.log_sysnoise = torch.nn.Parameter(torch.tensor(log_sysnoise, device="cuda"))#torch.tensor(log_sysnoise)
        self.log_obsnoise = torch.nn.Parameter(torch.tensor(log_obsnoise, device="cuda"))
        '''
        self.log_sysnoise = torch.tensor(log_sysnoise, device="cuda")
        self.log_obsnoise = torch.tensor(log_obsnoise, device="cuda")
        #self.Q = torch.exp(2 * self.log_sysnoise) * torch.eye(z_dim, device="cuda")
        #self.R = torch.exp(2 * self.log_obsnoise) * torch.eye(x_dim, device=device)
        self.G_val = G_val
        self.time_series_input = time_series_input
        if self.time_series_input:
            self.time_series_length = time_series_length
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.architecture = architecture
        #assert (
        #    self.z_dim % 2 == 0
        #), "z_dim should be even numbered if we do not consider real-valued eigenvalues"
        self.m = 0 * torch.ones(self.z_dim).cuda()#to(device)
        self.u = torch.zeros(self.z_dim).cuda()#to(device)
        self.test_every = 1000
        self.aux_alpha = aux_alpha
        self.save_folder = save_folder
        self.periodic = periodic
        self.periodic_indices = periodic_indices
        self.period = period
        if log_concentration_periodic is not None:
            self.log_concentration_periodic = torch.nn.Parameter(torch.tensor(log_concentration_periodic, device="cuda"))
        else:
            self.log_concentration_periodic = None

        self.variable_kernelsize = variable_kernelsize
        self.kernel = kernel
        self.G_nondiag = G_nondiag
        self.obs_mode = obs_mode
        assert obs_mode in ["range_independent", "range_combined"]
        self.network_init(mode=self.mode)
        self.unitmatrix = unitmatrix
        self.sigma_block_diag = sigma_block_diag
        self.h_network_fixed_params = h_network_fixed_params
        if CZV is not None:
            self.CZV = torch.nn.Parameter(torch.tensor(CZV, device="cuda"))
        else:
            self.CZV = None
        if CkDP is not None:
            self.CkDP = torch.tensor(CkDP, device="cuda")
            #self.CkDP = torch.nn.Parameter(torch.tensor(CkDP, device="cuda"))
        else:
            self.CkDP = None
        if ClAH is not None:
            self.ClAH = torch.nn.Parameter(torch.tensor(ClAH, device="cuda"))
        else:
            self.ClAH = None
        if ClAV is not None:
            self.ClAV = torch.nn.Parameter(torch.tensor(ClAV, device="cuda"))
        else:
            self.ClAV = None
        if lambda_AH is not None:
            self.lambda_AH = torch.nn.Parameter(torch.tensor(lambda_AH, device="cuda"))
        else:
            self.lambda_AH = None
        if lambda_AV is not None:
            self.lambda_AV = torch.nn.Parameter(torch.tensor(lambda_AV, device="cuda"))
        else:
            self.lambda_AV = None
        if PhiDP_offset_short is not None:
            self.PhiDP_offset_short = torch.nn.Parameter(torch.tensor(PhiDP_offset_short, device="cuda"))
            #self.PhiDP_offset_short = torch.tensor(PhiDP_offset_short, device="cuda")
        else:
            self.PhiDP_offset_short = None
        if PhiDP_offset_long is not None:
            self.PhiDP_offset_long = torch.nn.Parameter(torch.tensor(PhiDP_offset_long, device="cuda"))
            #self.PhiDP_offset_long = torch.tensor(PhiDP_offset_long, device="cuda")
        else:
            self.PhiDP_offset_long = None
        self.mu_0 = torch.zeros(z_dim)
        self.V_init = 1e4 * torch.eye(z_dim).cuda()#to(device)
        self.V = 1e2 * torch.eye(self.z_dim).cuda()#to(device)
        self.V_inv = torch.inverse(self.V)
        self.m = 0 * torch.ones(self.z_dim).cuda()#to(device)
        self.u = torch.zeros(self.z_dim).cuda()#to(device)
        if self.mode in ["MPPAWR_DSD_6dim"]:
            #self.Q = torch.diag_embed(torch.exp(2 * torch.concat([self.log_sysnoise, -100*torch.ones(3, device="cuda")])))# * torch.eye(z_dim, device="cuda")
            self.Q = torch.diag_embed(torch.exp(self.log_sysnoise))
        elif self.mode in ["MPPAWR_DSD"]:
            self.Q = torch.diag_embed(torch.exp(self.log_sysnoise))

        self.Dynamics_allsteps = Dynamics_allsteps

    def network_init(self, mode):
        assert mode in [
            "MPPAWR_DSD",
            "MPPAWR_DSD_6dim",
        ]
        self.mode = mode
        if mode in ["MPPAWR_DSD_6dim"]:
            if self.obs_mode == "range_independent":
                self.f_network, self.G_network = network_init_MPPAWR_DSD(
                    z_dim=self.z_dim, x_dim=self.x_dim
                )
            elif self.obs_mode == "range_combined":
                self.f_network, self.G_network = network_init_MPPAWR_DSD_range_combined_6dim(
                    z_dim=self.z_dim, x_dim=self.x_dim, architecture=self.architecture, steps=800
                )
            self.train_F = False
            assert self.F is not None
            assert self.h_network is not None
        else:
            pass
    

    def compute_f_and_G(
        self,
        o_t,
    ):
        # input_dim = bs, xdim, nstep
        #print(f"{self.f_network(o_t)[0]=}")
        #print(f"{self.f_network(o_t)[0, 0]=}")
        #print(f"{self.f_network(o_t)[0, 1]=}")
        f_output = self.f_network(o_t)
        G_output = self.G_network(o_t)
        #print(f"{self.f_network(o_t)[0].shape=}")
        f_attenuation = (torch.atan(f_output[..., 0])+torch.pi/2)*4
        f_array = rearrange(f_output[..., 1:], "bs (zdim nstep) -> bs nstep zdim", zdim=3) # n0, mu, lambda
        #G_attenuation_ = torch.atan(G_output[..., 0]) # for now
        G_attenuation = torch.zeros_like(G_output[..., 0]) # for now #G_attenuation_*G_attenuation_
        G_array_ = rearrange(G_output[..., 1:], "bs (zdim nstep) -> bs nstep zdim", zdim=3) # n0, mu, lambda
        #f_array = rearrange(self.f_network(o_t), "bs zdim nstep -> bs nstep zdim") # bs, nstep, z_dim or z_dim//2
        #G_array_ = rearrange(self.G_network(o_t), "bs zdim nstep -> bs nstep zdim") # bs, nstep, z_dim or z_dim//2
        G_squared = G_array_ * G_array_  # (bs, nstep, z_dim or z_dim//2)
        return f_array, G_squared, f_attenuation, G_attenuation

    def _compute_mu_t_sigma_t(self, obs_data):
        o_t_all = rearrange(obs_data, "bs nstep xdim -> bs xdim nstep")
        ft_all, Gt_all, f_attenuation, G_attenuation = self.compute_f_and_G(
            o_t=o_t_all,
        ) # should contain means and stds of N0, mu, lambda
        for_sigmat = torch.concat([Gt_all, 1e-2*torch.ones_like(Gt_all)], dim=-1)
        mu_t_list_all = self.Dynamics_allsteps(z_half=ft_all, **self.h_network_fixed_params)
        sigma_t_list_all = torch.diag_embed(for_sigmat)
        return mu_t_list_all, sigma_t_list_all, f_attenuation, G_attenuation
    
    #def compute_predictions(self, obs_data, n_step, m_step, block_step, jump_step):
    def forward(self, obs_data, n_step, m_step, block_step, jump_step):
        ft_all, Gt_inv_all, f_attenuation, G_attenuation = self._compute_mu_t_sigma_t(
            obs_data=obs_data
        )
        #print(f"{ft_all.dtype=}")
        Gt_inv_all = Gt_inv_all + 1e-4*torch.eye(self.z_dim, device="cuda")
        #print(f"{ft_all.shape=}")
        #print(f"{Gt_inv_all.shape=}")
        #ft_all = rearrange(ft_all_, "bs xdim nstep-> bs nstep xdim")
        #Gt_inv_all = rearrange(Gt_inv_all_, "bs xdim nstep-> bs nstep xdim")
        h_results, h_attenuation = self.reparametrize(ft_all, Gt_inv_all, f_attenuation, G_attenuation)
        modification_short = torch.zeros_like(h_results)
        modification_short[:, 0, :118, 2] = 1.0
        modification_long = torch.zeros_like(h_results)
        modification_long[:, 0, 118:, 2] = 1.0
        h_results_modified = h_results + self.PhiDP_offset_short*modification_short + self.PhiDP_offset_long*modification_long

        return (
            ft_all,
            Gt_inv_all,
            h_results_modified,
            self.log_obsnoise,
            self.log_sysnoise,
            h_attenuation
        )

    def reparametrize(self, mu_1, sigma_1, mu_attenuation, sigma_attenuation): # works only for sigma_block_diag==True
        (N_data, n_step, z_dim) = mu_1.shape
        #sigma_1_symmetrized = 0.5 * (sigma_1 + sigma_1.transpose(3, 4))
        #sqrt_sigma_1 = torch.linalg.cholesky(sigma_1_symmetrized)
        sqrt_sigma_1 = torch.linalg.cholesky(sigma_1)
        mvn1 = dist.Normal(torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda"))
        N_sample = 1
        points = mvn1.sample((N_data, n_step, N_sample, z_dim))
        result = mu_1.unsqueeze(2).repeat(1, 1, N_sample, 1) + torch.einsum(
            "abij,absj->absi", sqrt_sigma_1, points
        )  # (N_data, n_step, N_sample, z_dim)
        h_arg = result.transpose(1, 2) # (N_data, N_sample, n_step, z_dim)
        h_arg = rearrange(h_arg, "d s st z-> (d s st) z")
        #print(f"{mu_attenuation.shape=}")
        #print(f"{sigma_attenuation.shape=}")
        #print(f"{mu_attenuation.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,N_sample,n_step,1).shape=}")
        #print(f"{mvn1.sample((N_data,1,n_step,1)).shape=}")
        h_attenuation_ = mu_attenuation.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,N_sample,n_step,1) + torch.einsum("a,abcd->abcd", sigma_attenuation, mvn1.sample((N_data,1,n_step,1)))
        h_attenuation = rearrange(h_attenuation_, "d s st z -> (d s st) z")
        if self.CkDP is None:
            h_results_ = self.h_network(h_arg)
        else: # for MPPAWR
            h_results_ = self.h_network(h_arg, h_attenuation, **self.h_network_fixed_params)
        h_results = rearrange(h_results_, "(d s st) x -> d s st x", d=N_data, s=N_sample, st=n_step)
        return h_results, h_attenuation
