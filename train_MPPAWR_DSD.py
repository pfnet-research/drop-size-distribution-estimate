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
train_data_path = "train_data"

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
    ZDR = Zdr - (PIAH-PIAV)
    return torch.stack([ZH, ZDR, PhiDP], dim=-1)

                        
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size):
    ddp_setup(rank, world_size)
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
    ZH_file_path_list_train = []
    ZDR_file_path_list_train = []
    PHIDP_file_path_list_train = []
    RHOHV_file_path_list_train = []
    ZH_file_path_list_test = []
    ZDR_file_path_list_test = []
    PHIDP_file_path_list_test = []
    RHOHV_file_path_list_test = []

    datetimes = np.array([])
    #filelist = ["202105", "202106", "202107", "202108", "202109"]
    filelist = ["202106", "202107", "202206"]
    for data in filelist:
        datetime = np.loadtxt(f"{txtfiles_datetime_path}/{data}_datetime_10MB.txt", dtype=object)
        datetimes = np.concatenate([datetimes, datetime])

    num_data = 4*int(len(datetimes)//4)
    datetimes = datetimes[:num_data]
        
    for datetime in datetimes:
        print(f"{datetime=}")
        train_year = datetime[:4]
        train_month = datetime[4:6]
        train_day = datetime[6:8]
        train_hour = datetime[9:11]
        ZH_file_path_list_train_tmp = [f"{train_data_path}/{train_year}/{train_month}/{train_day}/{train_hour}/{datetime}.00-00-PPI.RAW-ZH_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        ZDR_file_path_list_train_tmp = [f"{train_data_path}/{train_year}/{train_month}/{train_day}/{train_hour}/{datetime}.00-00-PPI.RAW-ZDR_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        PHIDP_file_path_list_train_tmp = [f"{train_data_path}/{train_year}/{train_month}/{train_day}/{train_hour}/{datetime}.00-00-PPI.RAW-PHIDP_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        RHOHV_file_path_list_train_tmp = [f"{train_data_path}/{train_year}/{train_month}/{train_day}/{train_hour}/{datetime}.00-00-PPI.RAW-RHOHV_MTI.AUTO-02-NORMAL.saitama.dat.gz"]
        ZH_file_path_list_train += ZH_file_path_list_train_tmp
        ZDR_file_path_list_train += ZDR_file_path_list_train_tmp
        PHIDP_file_path_list_train += PHIDP_file_path_list_train_tmp
        RHOHV_file_path_list_train += RHOHV_file_path_list_train_tmp
        
    N_data = len(ZH_file_path_list_train)
    #N_data_test = len(ZH_file_path_list_test)
    print(f"{N_data=}")
    print(f"{ZH_file_path_list_train[0]=}")
    
    dataset = MPPAWR_DSD(
        num_data=N_data,
        n_steps=steps_to_generate,
        ZH_file_path_list=ZH_file_path_list_train,
        ZDR_file_path_list=ZDR_file_path_list_train,
        PHIDP_file_path_list=PHIDP_file_path_list_train,
        RHOHV_file_path_list=RHOHV_file_path_list_train,
        savefolder=outdir,
        elevation_angles=elevation_angles
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=sampler, pin_memory=True)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    dataloader_list = [dataloader]
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
    train_attenuation = False
    if train_attenuation:
        parameters.extend([
            {"params": model.ClAH},
            {"params": model.ClAV},
            {"params": model.lambda_AH},
            {"params": model.lambda_AV},
        ])
    model = DDP(model, device_ids=[rank])

    i_loop = 0
    # training loop.
    for i_loop in range(num_epochs):
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
        print(f"{i_loop=}/num_epochs")
        if i_loop == 0:
            optimizer = torch.optim.Adam(parameters, lr=lr)
            scheduler = ExponentialLR(optimizer, gamma=gamma_per_iter)
            savename = os.path.join(outdir, "fhKGR")
        for batch_idx, batch in tenumerate(dataloader):
            # train mode
            ZH_data = batch[0][0]
            #print(f"{torch.max(ZH_data)=}, {torch.min(ZH_data)=}")
            ZDR_data = batch[1][0]
            PHIDP_data = batch[2][0]
            RHOHV_data = batch[3][0]#.to("cuda")
            angle_offset = batch[4][0]
            #print(f"{angle_offset=}")
            #print(f"{ZH_data.dim()=}")
            if ZH_data.dim() == 0:
                print("skip data")
                continue
            '''
            torch.save(ZH_data, os.path.join(outdir, "ZH_data"))
            torch.save(ZDR_data, os.path.join(outdir, "ZDR_data"))
            torch.save(PHIDP_data, os.path.join(outdir, "PHIDP_data"))
            torch.save(RHOHV_data, os.path.join(outdir, "RHOHV_data"))
            '''
            #print(f"{ZH_data.shape=}")
            #print(f"{ZDR_data.shape=}")
            #print(f"{PHIDP_data.shape=}")
            #print(f"{RHOHV_data.shape=}")
            obs_data = torch.stack([ZH_data, ZDR_data, PHIDP_data, RHOHV_data], dim=-1).to("cuda")
            obs_data_normed = torch.stack([ZH_data/50, ZDR_data, PHIDP_data/20, RHOHV_data], dim=-1).to("cuda") # multiply 50 on ZDR_data to make it more prominent
            #print(f"{obs_data.shape=}")
            target = obs_data[..., :3]
            optimizer.zero_grad()
            (
                mu_t_list_all,
                sigma_t_list_all,
                h_results,
                log_obsnoise,
                log_sysnoise,
                h_attenuation,
            ) = model(obs_data=obs_data_normed, n_step=n_step, m_step=m_step, block_step=1, jump_step=1)
            '''
            print(f"{mu_t_list_all[..., 0]=}")
            print(f"{mu_t_list_all[..., 1]=}")
            print(f"{mu_t_list_all[..., 2]=}")
            print(f"{mu_t_list_all[..., 3]=}")
            print(f"{mu_t_list_all[..., 4]=}")
            print(f"{mu_t_list_all[..., 5]=}")
            '''
            #print(f"{mu_t_list_all=}")
            #print(f"{sigma_t_list_all=}")
            jump_step_loss = 1
            loss_KL = compute_KL_sampling(
                mu_1=mu_t_list_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_list_all[:, ::jump_step_loss, :, :],
                sigma_2=model.module.Q,
                dynamics=Dynamics,
                z_dim=z_dim,
                N_data=len(obs_data),
                n_step=np.ceil((n_step + m_step)/jump_step_loss),

                dynamics_args={"N0_range": N0_range, "loc_range": loc_range, "log_scale_correction_range": log_scale_correction_range, "lookup_tables": lookup_tables},
            )
            '''
            loss_KL = compute_KL_gaussians(
                mu_1=mu_t_list_all[:, ::jump_step_loss, :],
                sigma_1=sigma_t_list_all[:, ::jump_step_loss, :, :],
                mu_2=mu_t_p_list_all[:, ::jump_step_loss, :],
                sigma_2=sigma_t_p_list_all[:, ::jump_step_loss, :, :],
                z_dim=z_dim,
                N_data=len(obs_data),
                n_step=np.ceil((n_step + m_step)/jump_step_loss),
                sigma_block_diag=False,
            )
            '''
            '''
            h_results, PIAH, PIAV = attenuation_correction_DSD(h_results, mu_t_list_all,
                                                               lambda_AH=model.module.lambda_AH,
                                                               lambda_AV=model.module.lambda_AV,
                                                               ClAH=model.module.ClAH,
                                                               ClAV=model.module.ClAV)
            '''
            
            #modification_short = torch.zeros_like(h_results)
            #modification_short[:, 0, :118, 2] = 1.0
            #modification_long = torch.zeros_like(h_results)
            #modification_long[:, 0, 118:, 2] = 1.0
            
            #h_results_modified = h_results + model.PhiDP_offset_short*modification_short + model.PhiDP_offset_long*modification_long#Zdr_and_PhiDP(h_results, model.module.PhiDP_offset)
            #mask1 = torch.stack([torch.ones_like(RHOHV_data), (RHOHV_data>0.95), (RHOHV_data>0.95)], dim=-1).to("cuda") # RHOHV cond, applied to Zdr and PhiDP
            target[..., 1] = torch.where(RHOHV_data.to("cuda")>0.95, target[..., 1], 0.3)
            mask1 = torch.stack([torch.ones_like(RHOHV_data).to("cuda"), (RHOHV_data.to("cuda")>0.95)+(RHOHV_data.to("cuda")<=0.95)*(target[..., 0]<10), (RHOHV_data.to("cuda")>0.95)*(target[..., 0]>10)], dim=-1).to("cuda") # RHOHV cond, applied to Zdr and PhiDP
            mask2 = (target>-20) # nan cond, [6000, 400, 3]
            mask_all = mask1*mask2
            loss_integral, loss0, loss1, loss2 = compute_loss_integral5(
                h_results=h_results,
                x_value=target,
                sigma_err=torch.exp(log_obsnoise),
                mask=mask_all,
            )
            loss = loss_KL - loss_integral
            #loss = -loss_integral

            loss_integral_list.append(loss_integral.detach().cpu().numpy())
            loss0_list.append(loss0.detach().cpu().numpy())
            loss1_list.append(loss1.detach().cpu().numpy())
            loss2_list.append(loss2.detach().cpu().numpy())
            loss_KL_list.append(loss_KL.detach().cpu().numpy())
            log_obsnoise_list.append(model.module.log_obsnoise.detach().cpu().numpy())
            log_sysnoise_list.append(model.module.log_sysnoise.detach().cpu().numpy())
            CkDP_list.append(model.module.CkDP.detach().cpu().numpy())
            PhiDP_offset_short_list.append(model.module.PhiDP_offset_short.detach().cpu().numpy())
            PhiDP_offset_long_list.append(model.module.PhiDP_offset_long.detach().cpu().numpy())
            last_lr_list.append(scheduler.get_last_lr())

            
            #print(f"{loss=}, {loss_KL=}, {loss_integral=}")

            if batch_idx % 20 == 0:
                print(f"{loss=}, {loss_KL=}, {loss_integral=}")
                diff = h_results-target.unsqueeze(1)
                diff_masked_0 = diff[..., 0][mask_all.unsqueeze(1)[..., 0] != 0].flatten()
                diff_masked_1 = diff[..., 1][mask_all.unsqueeze(1)[..., 1] != 0].flatten()
                diff_masked_2 = diff[..., 2][mask_all.unsqueeze(1)[..., 2] != 0].flatten()
                print(f"{diff_masked_2=}")
                print(f"{torch.mean(diff_masked_0*diff_masked_0)=}, {len(diff_masked_0)=}, {loss0.detach()=}\n")
                print(f"{torch.mean(diff_masked_1*diff_masked_1)=}, {len(diff_masked_1)=}, {loss1.detach()=}\n")
                print(f"{torch.mean(diff_masked_2*diff_masked_2)=}, {len(diff_masked_2)=}, {loss2.detach()=}\n")
                print(f"{model.module.PhiDP_offset_long=}")
                print(f"{model.module.PhiDP_offset_short=}")
                print(f"{torch.mean(h_attenuation)=}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1000)
            '''
            for n, p in model.f_network.named_parameters():
                print(f"{n=}, {torch.max(p.grad)=}, {torch.min(p.grad)=}, {torch.std(p.grad)=}")
            '''
            #model.module.Q = torch.diag_embed(torch.exp(2 * torch.concat([model.module.log_sysnoise, -100*torch.ones(3, device="cuda")])))
            '''
            Q_nondiag = torch.zeros(z_dim, z_dim, device="cuda")
            Q_nondiag[1, 2] = torch.exp(0.5*(model.log_sysnoise[1]+model.log_sysnoise[2]))*0.9 # correlation between mu and lambda is 0.8
            Q_nondiag[2, 1] = Q_nondiag[1, 2]
            '''
            model.module.Q = torch.diag_embed(torch.exp(model.module.log_sysnoise))# + Q_nondiag
            optimizer.step()
            scheduler.step()
            
            #print(f"{model.module.CkDP=}")
            #print(f"{model.module.PhiDP_offset=}")
            #if ((i_loop % test_every == 0) and (batch_idx == 0)): # test, only when i_loop % test_every == 0 and before batch_idx==0
        np.savetxt(os.path.join(outdir, f"trainloss_integral_epoch{i_loop}_rank{rank}.txt"), np.array(loss_integral_list))
        np.savetxt(os.path.join(outdir, f"trainloss_integral0_epoch{i_loop}_rank{rank}.txt"), np.array(loss0_list))
        np.savetxt(os.path.join(outdir, f"trainloss_integral1_epoch{i_loop}_rank{rank}.txt"), np.array(loss1_list))
        np.savetxt(os.path.join(outdir, f"trainloss_integral2_epoch{i_loop}_rank{rank}.txt"), np.array(loss2_list))
        np.savetxt(os.path.join(outdir, f"trainloss_KL_epoch{i_loop}_rank{rank}.txt"), np.array(loss_KL_list))
        np.savetxt(os.path.join(outdir, f"log_obsnoise_epoch{i_loop}_rank{rank}.txt"), np.array(log_obsnoise_list))
        np.savetxt(os.path.join(outdir, f"log_sysnoise_epoch{i_loop}_rank{rank}.txt"), np.array(log_sysnoise_list))
        #np.savetxt(os.path.join(outdir, "CkDP.txt"), np.array(CkDP_list))
        np.savetxt(os.path.join(outdir, f"PhiDP_offset_short_epoch{i_loop}_rank{rank}.txt"), np.array(PhiDP_offset_short_list))
        np.savetxt(os.path.join(outdir, f"PhiDP_offset_long_epoch{i_loop}_rank{rank}.txt"), np.array(PhiDP_offset_long_list))
        np.savetxt(os.path.join(outdir, f"lr_epoch{i_loop}_rank{rank}.txt"), np.array(last_lr_list))

    # final updates
    log_obsnoise_list.append(model.module.log_obsnoise.detach().cpu().numpy())
    log_sysnoise_list.append(model.module.log_sysnoise.detach().cpu().numpy())
    #CkDP_list.append(model.CkDP.detach().cpu().numpy())
    PhiDP_offset_short_list.append(model.module.PhiDP_offset_short.detach().cpu().numpy())
    PhiDP_offset_long_list.append(model.module.PhiDP_offset_long.detach().cpu().numpy())
    last_lr_list.append(scheduler.get_last_lr())
    
    # test
    #torch.save(model.CkDP, os.path.join(outdir, f"{savename}_test_CkDP"))
    torch.save(model.module.PhiDP_offset_short, os.path.join(outdir, f"{savename}_test_model_PhiDP_offset_short"))
    torch.save(model.module.PhiDP_offset_long, os.path.join(outdir, f"{savename}_test_model_PhiDP_offset_long"))
    torch.save(model.module.f_network.state_dict(), os.path.join(outdir, f"{savename}_f_network_state_dict"))
    torch.save(model.module.G_network.state_dict(), os.path.join(outdir, f"{savename}_G_network_state_dict"))
        
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"{world_size=}")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    #main()

