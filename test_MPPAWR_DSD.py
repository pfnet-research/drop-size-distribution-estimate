import argparse
import torch.multiprocessing as mp
import numpy as np
import os
import torch
from DSD.datasets.MPPAWR_DSD import MPPAWR_DSD
from DSD.models.direct_DSD_radomeattenuation import Direct_DDP, compute_KL_gaussians, compute_loss_integral5, compute_KL_sampling
from read_config import read_config
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR
import math
from tqdm.contrib import tenumerate

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lookup_tables_path = "lookup_tables"
txtfiles_datetime_path = "txtfiles_datetime"
test_data_path = "test_data"
model_weight_path = "model_weights/model_weight"

def torch_interp1d(x, xp, yp):
    #print(f"{x.device=}, {xp.device=}")
    idx_left = torch.searchsorted(xp, x, right=True) - 1
    idx_right = idx_left + 1
    slopes = (yp[idx_right] - yp[idx_left]) / (xp[idx_right] - xp[idx_left])
    interpolated = yp[idx_left] + slopes * (x - xp[idx_left])
    return interpolated

def torch_interp2d(x, y, xp, yp, zp):
    #print(f"{x.device=}, {xp.device=}")
    idx_left = torch.searchsorted(xp, x, right=True) - 1
    idx_right = idx_left + 1
    idx_lower = torch.searchsorted(yp, y, right=True) - 1
    idx_upper = idx_lower + 1
    w_x = (x-xp[idx_left])/(xp[idx_right]-xp[idx_left])
    w_y = (y-yp[idx_lower])/(yp[idx_upper]-yp[idx_lower])
    interpolated = w_x*w_y*zp[idx_right, idx_upper] + (1-w_x)*w_y*zp[idx_left, idx_upper] + w_x*(1-w_y)*zp[idx_right, idx_lower] + (1-w_x)*(1-w_y)*zp[idx_left, idx_lower]
    return interpolated

def load_lookups(el):
    table_AH = torch.tensor(np.loadtxt(f"{lookup_tables_path}/lognorm_AH_el{el}.txt"), device=device, dtype=torch.float32)
    table_AV = torch.tensor(np.loadtxt(f"{lookup_tables_path}/lognorm_AV_el{el}.txt"), device=device, dtype=torch.float32)
    table_kdp = torch.tensor(np.loadtxt(f"{lookup_tables_path}/lognorm_kdp_el{el}.txt"), device=device, dtype=torch.float32)
    table_ZH = torch.tensor(np.loadtxt(f"{lookup_tables_path}/lognorm_ZH_el{el}.txt"), device=device, dtype=torch.float32)
    table_Zdr = torch.tensor(np.loadtxt(f"{lookup_tables_path}/lognorm_Zdr_el{el}.txt"), device=device, dtype=torch.float32)
    return [table_AH*0.075*2, table_AV*0.075*2, table_kdp*0.075*2, table_ZH, table_Zdr]

def Maki(x):
    x_cm = x * 0.1
    # Apply conditions for both cases
    condition = (x_cm < 0.11) | (x_cm > 0.44)
    # Define the two cases
    result_case1 = 1.0048 + 0.0057*x_cm - 2.628*x_cm**2 + 3.682*x_cm**3 - 1.677*x_cm**4
    result_case2 = 1.0048 + 0.0057*x_cm - 2.628*x_cm**2 + 3.682*x_cm**3 - 1.677*x_cm**4
    # Use np.where to choose the correct case
    return torch.where(condition, result_case1, result_case2)

def Dynamics_allsteps(z_half, N0_range, loc_range, log_scale_correction_range, lookup_tables):
    [table_AH, table_AV, table_kdp, table_ZH, table_Zdr] = lookup_tables
    N0s = torch.sigmoid(z_half[..., 0:1]/(N0_range[1]-N0_range[0]))*(N0_range[1]-N0_range[0])+N0_range[0]
    locs = torch.sigmoid(z_half[..., 1:2]/(loc_range[1]-loc_range[0]))*(loc_range[1]-loc_range[0])+loc_range[0]
    log_scale_corrections = torch.sigmoid(z_half[..., 2:3]/(log_scale_correction_range[1]-log_scale_correction_range[0]))*(log_scale_correction_range[1]-log_scale_correction_range[0])+log_scale_correction_range[0]
    xp = torch.linspace(-1.2, 1.2, 61, device=z_half.device)
    yp = torch.linspace(-0.2, 0.2, 51, device=z_half.device)
    Zh = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_ZH) + 10*N0s
    Zdr = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_Zdr)
    kDP = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_kdp) * pow(10, N0s)
    AH = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_AH) * pow(10, N0s)
    AV = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_AV) * pow(10, N0s)
    PIAH = torch.cumsum(AH, dim=-2)
    PIAV = torch.cumsum(AV, dim=-2)
    PhiDP = torch.cumsum(kDP, dim=-2)
    return torch.concat([z_half[..., 0:1], z_half[..., 1:2], z_half[..., 2:3], PhiDP, PIAH, PIAV], dim=-1)

def Dynamics(z, N0_range, loc_range, log_scale_correction_range, lookup_tables):
    [table_AH, table_AV, table_kdp, table_ZH, table_Zdr] = lookup_tables
    N0s = torch.sigmoid(z[..., 0]/(N0_range[1]-N0_range[0]))*(N0_range[1]-N0_range[0])+N0_range[0]
    locs = torch.sigmoid(z[..., 1]/(loc_range[1]-loc_range[0]))*(loc_range[1]-loc_range[0])+loc_range[0]
    log_scale_corrections = torch.sigmoid(z[..., 2]/(log_scale_correction_range[1]-log_scale_correction_range[0]))*(log_scale_correction_range[1]-log_scale_correction_range[0])+log_scale_correction_range[0]

    xp = torch.linspace(-1.2, 1.2, 61, device=z.device)
    yp = torch.linspace(-0.2, 0.2, 51, device=z.device)
    Zh = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_ZH) + 10*N0s
    Zdr = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_Zdr)
    kDP = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_kdp) * pow(10, N0s)
    AH = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_AH) * pow(10, N0s)
    AV = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_AV) * pow(10, N0s)
    PhiDP = z[..., 3]
    PIAH = z[..., 4]
    PIAV = z[..., 5]
    return torch.stack([z[..., 0], z[..., 1], z[..., 2], PhiDP+kDP, PIAH+AH, PIAV+AV], dim=-1)

def observation_DSD_Zh(z, z_attenuation, N0_range, loc_range, log_scale_correction_range, lookup_tables):
    #print(f"{z.shape=}")
    #print(f"{z_attenuation.shape=}")
    [table_AH, table_AV, table_kdp, table_ZH, table_Zdr] = lookup_tables
    # assuming z contains 3 numbers per ot. 3 numbers are ZH, ZV, KDP
    N0s = torch.sigmoid(z[..., 0]/(N0_range[1]-N0_range[0]))*(N0_range[1]-N0_range[0])+N0_range[0]
    locs = torch.sigmoid(z[..., 1]/(loc_range[1]-loc_range[0]))*(loc_range[1]-loc_range[0])+loc_range[0]
    log_scale_corrections = torch.sigmoid(z[..., 2]/(log_scale_correction_range[1]-log_scale_correction_range[0]))*(log_scale_correction_range[1]-log_scale_correction_range[0])+log_scale_correction_range[0]

    xp = torch.linspace(-1.2, 1.2, 61, device=z.device)
    yp = torch.linspace(-0.2, 0.2, 51, device=z.device)
    Zh = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_ZH) + 10*N0s
    Zdr = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_Zdr)
    kDP = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_kdp) * pow(10, N0s)
    AH = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_AH) * pow(10, N0s)
    AV = torch_interp2d(x=locs, y=log_scale_corrections, xp=xp, yp=yp, zp=table_AV) * pow(10, N0s)

    PhiDP = z[..., 3]
    PIAH = z[..., 4]
    PIAV = z[..., 5]

    ZH = Zh - PIAH - z_attenuation[..., 0]
    #print(f"{z_attenuation[0:10, 0]=}")
    ZDR = Zdr - (PIAH-PIAV)
    return torch.stack([ZH, ZDR, PhiDP], dim=-1)
                        
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main():
    #ddp_setup(rank, world_size)
    rank = 0
    world_size = 1
    parser = argparse.ArgumentParser(
        description="DSD estimate in MPPAWR observations"
    )

    parser.add_argument("--config", type=str, help="config file path", required=True)
    args = parser.parse_args()
    config_file = args.config
    config = read_config(f"{config_file}")

    variables = config["model"]["variables"]
    assert variables in ["DSD", "radar"]
    z_dim = int(config["model"]["z_dim"])
    log_sysnoise = float(config["model"]["log_sysnoise"])
    log_obsnoise_ZH = float(config["model"]["log_obsnoise_ZH"])
    log_obsnoise_Zdr = float(config["model"]["log_obsnoise_Zdr"])
    log_obsnoise_PhiDP = float(config["model"]["log_obsnoise_PhiDP"])
    architecture = config["model"]["architecture"]
    assert architecture in ["linear", "conv", "transformer"]
    
    block_step = int(config["model"]["block_step"])
    aux_alpha = float(config["model"]["aux_alpha"])
    lr = float(config["model"]["lr"])
    time_sequence_length = int(config["model"]["time_sequence_length"])
    G_val = float(config["model"]["G_val"])
    batch_size = int(config["model"]["batch_size"])
    jump_step_loss_setting = config["model"]["jump_step_loss_setting"]
    variable_kernelsize = config.getboolean("model", "variable_kernelsize")
    kernel = int(config["model"]["kernel"])
    N0_range_low = float(config["model"]["N0_range_low"])
    N0_range_high = float(config["model"]["N0_range_high"])
    loc_range_low = float(config["model"]["loc_range_low"])
    loc_range_high = float(config["model"]["loc_range_high"])
    log_scale_correction_range_low = float(config["model"]["log_scale_correction_range_low"])
    log_scale_correction_range_high = float(config["model"]["log_scale_correction_range_high"])
    N0_range = [N0_range_low, N0_range_high]
    loc_range = [loc_range_low, loc_range_high]
    log_scale_correction_range = [log_scale_correction_range_low, log_scale_correction_range_high]
    lr_10percent_iters = float(config["model"]["lr_10percent_iters"])
    gamma_per_iter = pow(10, -1.0/lr_10percent_iters)
    assert jump_step_loss_setting in ["pattern1", "pattern2"], jump_step_loss_setting

    N_data = int(config["data"]["N_data"])
    data = config["data"]["data"]
    assert data in ["complete", "quad_capped_10", "quad_capped_100", "quad_capped_1000"], f"{data=}"
    obsnoise = float(config["data"]["obsnoise"])
    train_data_seed = int(config["data"]["train_data_seed"])
    test_data_seed = int(config["data"]["test_data_seed"])
    n_step = int(config["data"]["n_step"])
    m_step = int(config["data"]["m_step"])
    n_step_test = int(config["data"]["n_step_test"])
    m_step_test = int(config["data"]["m_step_test"])
    dt = float(config["data"]["dt"])
    elevation_angles = config["data"]["elevation_angles"]
    print(f"{elevation_angles=}")
    if elevation_angles == "4_6":
        load_el = 5
    lookup_tables = load_lookups(el=load_el)
    print(f"{lookup_tables[0].shape=}")
    outdir = config["others"]["outdir"]

    batch_size = 1
    dt = 1.0
    n_step = 800
    m_step = 0
    n_step_test = 800
    m_step_test = 0
    steps_to_generate = n_step + m_step
    z_dim = 6 # N0, mu, Lambda
    x_dim = 3 # ZH, Zdr, PhiDP
    input_dim = 45*800*76*3 # Vh, Zh, Zh_prev
    num_epochs = 20
    test_every = 1
    
    print("preparing data...")
    print(f"{data=}")
        
    # prepare dataloaders.

    # new datafiles
    ZH_file_path_list_test = []
    ZDR_file_path_list_test = []
    PHIDP_file_path_list_test = []
    RHOHV_file_path_list_test = []

    test_datetimes = np.array([])
    #filelist = ["202206", "202207"]                                                                                                            
    filelist = ["202206"]#, "202207"]
    for data in filelist:
        test_datetime = np.loadtxt(f"{txtfiles_datetime_path}/{data}_datetime_Kumagaya.txt", dtype=object)
        test_datetimes = np.concatenate([test_datetimes, test_datetime])
    print(f"{test_datetimes=}")
    print(f"{len(test_datetimes)=}")
    #test_datetimes = test_datetimes[1246:]

    for datetime in test_datetimes:
        print(f"{datetime=}")
        test_year = datetime[:4]
        test_month = datetime[4:6]
        test_day = datetime[6:8]
        test_hour = datetime[9:11]
        ZH_file_path_list_test_tmp = [f"{test_data_path}/{test_year}/{test_month}/{test_day}/{test_hour}/{datetime}.00-00-PPI.RAW-ZH_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        ZDR_file_path_list_test_tmp = [f"{test_data_path}/{test_year}/{test_month}/{test_day}/{test_hour}/{datetime}.00-00-PPI.RAW-ZDR_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        PHIDP_file_path_list_test_tmp = [f"{test_data_path}/{test_year}/{test_month}/{test_day}/{test_hour}/{datetime}.00-00-PPI.RAW-PHIDP_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        RHOHV_file_path_list_test_tmp = [f"{test_data_path}/{test_year}/{test_month}/{test_day}/{test_hour}/{datetime}.00-00-PPI.RAW-RHOHV_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        ZH_file_path_list_test += ZH_file_path_list_test_tmp
        ZDR_file_path_list_test += ZDR_file_path_list_test_tmp
        PHIDP_file_path_list_test += PHIDP_file_path_list_test_tmp
        RHOHV_file_path_list_test += RHOHV_file_path_list_test_tmp
        
    N_data_test = int(len(ZH_file_path_list_test))
    print(f"{N_data_test=}")
    print(f"{ZH_file_path_list_test[0]=}")
    testset = MPPAWR_DSD(
        num_data=N_data_test,
        n_steps=steps_to_generate,
        ZH_file_path_list=ZH_file_path_list_test[rank*N_data_test:(rank+1)*N_data_test],
        ZDR_file_path_list=ZDR_file_path_list_test[rank*N_data_test:(rank+1)*N_data_test],
        PHIDP_file_path_list=PHIDP_file_path_list_test[rank*N_data_test:(rank+1)*N_data_test],
        RHOHV_file_path_list=RHOHV_file_path_list_test[rank*N_data_test:(rank+1)*N_data_test],
        savefolder=outdir,
        elevation_angles=elevation_angles
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=16
    )
    print("dataloader ok")
    
    # prepare models.
    take_blockstep = False
    take_physical_loss = False
    mode = "MPPAWR_DSD_6dim"
    model = Direct_DDP(
        mode=mode,
        F=Dynamics,
        h_network=observation_DSD_Zh,
        take_loss_physical=take_physical_loss,
        take_blockstep=take_blockstep,
        log_sysnoise=[log_sysnoise, log_sysnoise, log_sysnoise-5, 10, 5, 5], # for PhiDP, PIAH, PIAV, we just put arbitrary number
        log_obsnoise=[log_obsnoise_ZH, log_obsnoise_Zdr, log_obsnoise_PhiDP],
        z_dim=z_dim,
        x_dim=x_dim,
        architecture=architecture,
        G_val=G_val,
        time_series_input=False,
        time_series_length=time_sequence_length,
        aux_alpha=aux_alpha,
        save_folder=outdir,
        variable_kernelsize=variable_kernelsize,
        sigma_block_diag=False,
        unitmatrix=False,
        G_nondiag=False,
        kernel=kernel,
        #CZV=1.5,
        CkDP=0.0,
        ClAH=None,
        ClAV=None,
        lambda_AH=None,
        lambda_AV=None,
        PhiDP_offset_short=20.,
        PhiDP_offset_long=20.,
        h_network_fixed_params={"N0_range": N0_range, "loc_range": loc_range, "log_scale_correction_range": log_scale_correction_range, "lookup_tables": lookup_tables},
        obs_mode="range_combined",
        f_nonlinear=True,
        Dynamics_allsteps=Dynamics_allsteps,
    )
    time_series_input = True
    model.log_obsnoise.requires_grad = True
    time_series_input = True
    model.log_obsnoise.requires_grad = True
    parameters = [
        {"params": model.f_network.parameters()},
        {"params": model.G_network.parameters()},
        {"params": model.PhiDP_offset_short},
        {"params": model.PhiDP_offset_long},
    ]

    model.f_network.load_state_dict(torch.load(f'{model_weight_path}/modelload2_fhKGR_f_network_state_dict', weights_only=False, map_location="cuda:0"))
    model.G_network.load_state_dict(torch.load(f'{model_weight_path}/modelload2_fhKGR_G_network_state_dict', weights_only=False, map_location="cuda:0"))
    model.PhiDP_offset_short = torch.load(f'{model_weight_path}/modelload2_fhKGR_test_model_PhiDP_offset_short', map_location="cuda:0")
    model.PhiDP_offset_long = torch.load(f'{model_weight_path}/modelload2_fhKGR_test_model_PhiDP_offset_long', map_location="cuda:0")
    print(f"{model.PhiDP_offset_short=}")
    print(f"{model.PhiDP_offset_long=}")
    print("weight loaded")
    model.eval()
    loss_integral_list = []
    loss0_list = []
    loss1_list = []
    loss2_list = []
    loss_KL_list = []
    log_obsnoise_list = []
    log_sysnoise_list = []
    CkDP_list = []
    PhiDP_offset_short_list = []
    PhiDP_offset_long_list = []
    last_lr_list = []
    with torch.no_grad():
        for batch_idx_test, batch_test in enumerate(testloader):
            ZH_data = batch_test[0][0]
            ZDR_data = batch_test[1][0]
            PHIDP_data = batch_test[2][0]
            RHOHV_data = batch_test[3][0]#.to("cuda")
            angle_offset = batch_test[4][0]
            if ZH_data.dim() == 0:
                print("skip data")
                continue
            obs_data = torch.stack([ZH_data, ZDR_data, PHIDP_data, RHOHV_data], dim=-1).to("cuda")
            obs_data_normed = torch.stack([ZH_data/50, ZDR_data, PHIDP_data/20, RHOHV_data], dim=-1).to("cuda")
            target = obs_data[..., :3]
            (
                mu_t_list_all,
                sigma_t_list_all,
                h_results,
                log_obsnoise,
                log_sysnoise,
                h_attenuation,
            ) = model(obs_data_normed, n_step, m_step, block_step=1, jump_step=1)
            jump_step_loss = 1
            loss_KL = compute_KL_sampling(
                mu_1=mu_t_list_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_list_all[:, ::jump_step_loss, :, :],
                sigma_2=model.Q,
                dynamics=Dynamics,
                z_dim=z_dim,
                N_data=len(obs_data),
                n_step=np.ceil((n_step + m_step)/jump_step_loss),
                dynamics_args={"N0_range": N0_range, "loc_range": loc_range, "log_scale_correction_range": log_scale_correction_range, "lookup_tables": lookup_tables},
            )
            target[..., 1] = torch.where(RHOHV_data.to("cuda")>0.95, target[..., 1], 0.3)
            mask1 = torch.stack([torch.ones_like(RHOHV_data).to("cuda"), (RHOHV_data.to("cuda")>0.95)+(RHOHV_data.to("cuda")<=0.95)*(target[..., 0]<10), (RHOHV_data.to("cuda")>0.95)*(target[..., 0]>10)], dim=-1).to("cuda") # RHOHV cond, applied to Zdr and PhiDP
            mask2 = (target>-20) # nan cond, [6000, 400, 3]
            mask_all = mask1*mask2
            
            #h_results_modified = h_results + model.PhiDP_offset_short*modification_short + model.PhiDP_offset_long*modification_long#Zdr_and_PhiDP(h_results, model.module.PhiDP_offset)
            diff = h_results-target.unsqueeze(1)
            print(f"iter {batch_idx_test}, {torch.mean(diff[..., 0]*diff[..., 0]*mask_all.unsqueeze(1)[..., 0]).detach().cpu().numpy()=}\n")
            with open(os.path.join(outdir, f"testdiff_ZH_rank{rank}.txt"), "a") as f:
                f.write(f"{torch.mean(diff[..., 0]*diff[..., 0]*mask_all.unsqueeze(1)[..., 0]).detach().cpu().numpy()}\n")
            with open(os.path.join(outdir, f"testdiff_ZDR_rank{rank}.txt"), "a") as f:
                f.write(f"{torch.mean(diff[..., 1]*diff[..., 1]*mask_all.unsqueeze(1)[..., 1]).detach().cpu().numpy()}\n")
            with open(os.path.join(outdir, f"testdiff_PhiDP_rank{rank}.txt"), "a") as f:
                f.write(f"{torch.mean(diff[..., 2]*diff[..., 2]*mask_all.unsqueeze(1)[..., 2]).detach().cpu().numpy()}\n")
                        
            loss_integral, loss0, loss1, loss2 = compute_loss_integral5(
                h_results=h_results,
                x_value=target,
                sigma_err=torch.exp(log_obsnoise),
                mask=mask_all,
            )
            loss = loss_KL - loss_integral

            loss_integral_list.append(loss_integral.detach().cpu().numpy())
            loss0_list.append(loss0.detach().cpu().numpy())
            loss1_list.append(loss1.detach().cpu().numpy())
            loss2_list.append(loss2.detach().cpu().numpy())
            loss_KL_list.append(loss_KL.detach().cpu().numpy())
        
            print(f"{mu_t_list_all.detach().cpu().numpy().dtype=}")
            savename = "test"
            np.savez_compressed(os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_mu_t_list_all"), mu_t_list_all[:, :, :3].detach().cpu().numpy())
            np.savez_compressed(os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_h_attenuation"), h_attenuation.detach().cpu().numpy())
            np.savez_compressed(
                os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_obsdata"),
                obs_data[:, :n_step_test+m_step_test, :].detach().cpu().numpy()
            )
            np.savez_compressed(
                os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_maskall"),
                mask_all.detach().cpu().numpy()
            )
            np.savez_compressed(
                os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_hresults"),
                h_results.detach().cpu().numpy()
            )
            np.savetxt(os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_angle_offset.txt"), np.array([angle_offset]))
            #torch.save(mu_t_list_all, os.path.join(outdir, f"{savename}_epoch{i_loop}_iter{batch_idx_test}_mu_t_list_all"))
            #torch.save(mu_t_list_all[:, :, :2].detach().cpu(), os.path.join(outdir, f"{savename}_epoch{i_loop}_iter{batch_idx_test}_mu_t_list_all_reduced"))
            '''
            torch.save(
            obs_data[:, :n_step_test+m_step_test, :], os.path.join(outdir, f"{savename}_iter{batch_idx_test}_obsdata")
            )
            torch.save(RHOHV_data, os.path.join(outdir, f"{savename}_iter{batch_idx_test}_RHOHV_data"))
            '''
            #torch.save(model.CkDP, os.path.join(outdir, f"{savename}_epoch{i_loop}_iter{batch_idx_test}_CkDP"))
            torch.save(model.PhiDP_offset_short, os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_model_PhiDP_offset_short"))
            torch.save(model.PhiDP_offset_long, os.path.join(outdir, f"{savename}_rank{rank}_iter{batch_idx_test}_model_PhiDP_offset_long"))
            
            #model.train() # get back to train mode
            #destroy_process_group()
    np.savetxt(os.path.join(outdir, f"testloss_integral_rank{rank}.txt"), np.array(loss_integral_list))
    np.savetxt(os.path.join(outdir, f"testloss_integral0_rank{rank}.txt"), np.array(loss0_list))
    np.savetxt(os.path.join(outdir, f"testloss_integral1_rank{rank}.txt"), np.array(loss1_list))
    np.savetxt(os.path.join(outdir, f"testloss_integral2_rank{rank}.txt"), np.array(loss2_list))
    np.savetxt(os.path.join(outdir, f"testloss_KL_rank{rank}.txt"), np.array(loss_KL_list))
    
if __name__ == "__main__":
    #world_size = torch.cuda.device_count()
    #print(f"{world_size=}")
    #mp.spawn(main, args=(world_size,), nprocs=world_size)
    main()

