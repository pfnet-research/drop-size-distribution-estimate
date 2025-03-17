import numpy as np
import os
import torch
from numpy.random import MT19937, RandomState, SeedSequence
from scipy.signal import convolve2d
from torch.utils.data import Dataset
import sys
from DSD.datasets.rawdata2 import RAWData
from einops import rearrange
import time
from url_opener import read_file
import io
#from LSSM.LSSM.mppawr.utils.rawdata2 import RAWData

cos_az_array_base = torch.tensor(np.load("DSD/datasets/cos_az_array.npy")[::4], dtype=torch.float32)#.to("cuda")
sin_az_array_base = torch.tensor(np.load("DSD/datasets/sin_az_array.npy")[::4], dtype=torch.float32)#.to("cuda")
# for elevation, we take 1, 3, 5, ..., 89 degrees
index_list = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38,
              42, 46, 48, 50, 52, 54, 56, 58, 60, 62,
              64, 66, 68, 70, 72, 74, 76, 78, 80, 82,
              84, 86, 88, 90, 92, 94, 96, 98, 100, 102,
              104, 106, 108, 110, 112]
cos_el_array_base = torch.tensor(np.load("DSD/datasets/cos_el_array.npy")[index_list], dtype=torch.float32)#.to("cuda")
sin_el_array_base = torch.tensor(np.load("DSD/datasets/sin_el_array.npy")[index_list], dtype=torch.float32)#.to("cuda")
rad_array_base = torch.tensor([(i+1)*300 for i in range(100)])#.to("cuda")
grid_size = (45, 100, 76)

def h_simple(z):
    return z

def h_doppler(z):
    cos_az_array = cos_az_array_base.to("cuda")#.to(z.device)
    sin_az_array = sin_az_array_base.to("cuda")#.to(z.device)
    cos_el_array = cos_el_array_base.to("cuda")#.to(z.device)
    sin_el_array = sin_el_array_base.to("cuda")#.to(z.device)
    z_reshaped = rearrange(z, "b (c x y z) -> b c x y z", c=3, x=45, y=100, z=76)
    #z.reshape(len(z), 3, 29, 200, 76)
    doppler_vel = (
        z_reshaped[:, 0, :, :, :]*cos_el_array[None, :, None, None]*cos_az_array[None, None, None, :] +
        z_reshaped[:, 1, :, :, :]*cos_el_array[None, :, None, None]*sin_az_array[None, None, None, :] +
        z_reshaped[:, 2, :, :, :]*sin_el_array[None, :, None, None]
    )
    doppler_vel = rearrange(doppler_vel, "b x y z -> b (x y z)")

    return doppler_vel

def evolve_density_semi_lagrangian_vectorized(rho, offset_el, offset_r, offset_az, mask_thresh):
    el_coords = torch.arange(45, device="cuda").float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    r_coords = torch.arange(100, device="cuda").float().unsqueeze(-1).unsqueeze(0).unsqueeze(0)
    az_coords = torch.arange(76, device="cuda").float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    #el_coords = torch.arange(45).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    #r_coords = torch.arange(100).float().unsqueeze(-1).unsqueeze(0).unsqueeze(0)
    #az_coords = torch.arange(76).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    el_back = el_coords - offset_el
    r_back = r_coords - offset_r
    az_back = az_coords - offset_az

    el_back = torch.clip(el_back, 0, grid_size[0] - 1)
    r_back = torch.clip(r_back, 0, grid_size[1] - 1)
    az_back = torch.clip(az_back, 0, grid_size[2] - 1)

    rho_new = trilinear_interpolation_vector(rho, el_back, r_back, az_back, mask_thresh)
    return rho_new

def trilinear_interpolation_vector(data, x, y, z, mask_thresh):
    # Compute the indices for the eight surrounding points
    interpolated_values = torch.zeros_like(data)
    for i in range(len(data)):
        i_floor = torch.floor(x[i]).long()
        j_floor = torch.floor(y[i]).long()
        k_floor = torch.floor(z[i]).long()
        i_ceil = i_floor + 1
        j_ceil = j_floor + 1
        k_ceil = k_floor + 1
        i_floor = torch.where(i_floor < 0, 0, i_floor)
        j_floor = torch.where(j_floor < 0, 0, j_floor)
        k_floor = k_floor % 75
        i_ceil = torch.where(i_ceil > 44, 44, i_ceil)
        j_ceil = torch.where(j_ceil > 99, 99, j_ceil)
        k_ceil = k_ceil % 75

        # Compute the interpolation coefficients
        dx = x[i] - i_floor.float()
        dy = y[i] - j_floor.float()
        dz = (z[i] - k_floor.float()) % 75
        dx_inv = 1 - dx
        dy_inv = 1 - dy
        dz_inv = 1 - dz

        assert torch.max(dx) <= 1
        assert torch.max(dy) <= 1
        assert torch.max(dz) <= 1
        assert torch.min(dx) >= 0
        assert torch.min(dy) >= 0
        assert torch.min(dz) >= 0

        c000 = data[i, i_floor, j_floor, k_floor]
        c001 = data[i, i_floor, j_floor, k_ceil]
        c010 = data[i, i_floor, j_ceil, k_floor]
        c011 = data[i, i_floor, j_ceil, k_ceil]
        c100 = data[i, i_ceil, j_floor, k_floor]
        c101 = data[i, i_ceil, j_floor, k_ceil]
        c110 = data[i, i_ceil, j_ceil, k_floor]
        c111 = data[i, i_ceil, j_ceil, k_ceil]

        cond = (c000 > -30).int() + (c001 > -30).int() + (c010 > -30).int() + (c011 > -30).int() + (c100 > -30).int() + (c101 > -30).int() + (c110 > -30).int() + (c111 > -30).int()
        mask = (cond >= mask_thresh) # mash_thresh or more points are observed

        interpolated_value = (
            dx_inv * dy_inv * dz_inv * c000 +
            dx_inv * dy_inv * dz * c001 +
            dx_inv * dy * dz_inv * c010 +
            dx_inv * dy * dz * c011 +
            dx * dy_inv * dz_inv * c100 +
            dx * dy_inv * dz * c101 +
            dx * dy * dz_inv * c110 +
            dx * dy * dz * c111
        )
        
        interpolated_values[i] = torch.where(mask, interpolated_value, -30) # -30 stands for NaN due to more than four (out of eight) nearest vertices are NaN

    return interpolated_values

def h_doppler_and_Zh(z, Zh_prev, mask_thresh):
    cos_az_array = cos_az_array_base.to("cuda")
    sin_az_array = sin_az_array_base.to("cuda")
    cos_el_array = cos_el_array_base.to("cuda")
    sin_el_array = sin_el_array_base.to("cuda")
    rad_array = rad_array_base.to("cuda")
    
    if Zh_prev is None:
        # only output doppler velocity
        return h_doppler(z)
    else:
        #print(f"{Zh_prev.shape=}")
        Zh_prev_reshaped = rearrange(Zh_prev, "b (x y z) -> b x y z", x=45, y=100, z=76)
        pass
    z_reshaped = rearrange(z, "b (c x y z) -> b c x y z", c=3, x=45, y=100, z=76) # vx, vy, vz at coordinate el, rad, az
    
    vr_reshaped = (
        z_reshaped[:, 0, :, :, :]*cos_el_array[None, :, None, None]*cos_az_array[None, None, None, :] +
        z_reshaped[:, 1, :, :, :]*cos_el_array[None, :, None, None]*sin_az_array[None, None, None, :] +
        z_reshaped[:, 2, :, :, :]*sin_el_array[None, :, None, None]
    ) # m/s
    vel_reshaped = (
        -z_reshaped[:, 0, :, :, :]*sin_el_array[None, :, None, None]*cos_az_array[None, None, None, :] +
        -z_reshaped[:, 1, :, :, :]*sin_el_array[None, :, None, None]*sin_az_array[None, None, None, :] +
        z_reshaped[:, 2, :, :, :]*cos_el_array[None, :, None, None]
    ) # m/s
    
    vaz_reshaped = (
        -z_reshaped[:, 0, :, :, :]*sin_az_array[None, None, None, :] +
        z_reshaped[:, 1, :, :, :]*cos_az_array[None, None, None, :]
    ) # m/s
    
    offset_r = vr_reshaped * 30 / 300 # cells
    offset_el = vel_reshaped * 30 / rad_array[None, None, :, None] * 180 / np.pi / 2 # cells
    offset_az = vaz_reshaped * 30 / rad_array[None, None, :, None] / cos_el_array[None, :, None, None] * 180 / np.pi / 4.8 # cells

    Zh_new = evolve_density_semi_lagrangian_vectorized(Zh_prev_reshaped, offset_el, offset_r, offset_az, mask_thresh)
    
    vr = rearrange(vr_reshaped, "b x y z -> b (x y z)")
    Zh = rearrange(Zh_new, "b x y z -> b (x y z)")
    #output = torch.cat([vr, Zh], axis=1)
    
    return vr, Zh #output

class MPPAWR_DSD(Dataset):
    def __init__(
        self,
        num_data,
        n_steps, # corresponds to the number of ranges
        ZH_file_path_list=[],
        ZDR_file_path_list=[],
        PHIDP_file_path_list=[],
        RHOHV_file_path_list=[],
        elevation_angles="",
        savefolder="",
    ):
        
        assert elevation_angles in ["0_2", "3_5", "6_8", "9_11",
                                "12_14", "15_17", "18_20", "21_23",
                                "24_26", "27_29", "30_32", "33_35",
                                    "36_38", "39_41", "42_44", "4_6"]
        if elevation_angles == "0_2":
            self.el1 = 0
            self.el2 = 3
        elif elevation_angles == "3_5":
            self.el1 = 3
            self.el2 = 6
        elif elevation_angles == "6_8":
            self.el1 = 6
            self.el2 = 9
        elif elevation_angles == "9_11":
            self.el1 = 9
            self.el2 = 12
        elif elevation_angles == "12_14":
            self.el1 = 12
            self.el2 = 15
        elif elevation_angles == "15_17":
            self.el1 = 15
            self.el2 = 18
        elif elevation_angles == "18_20":
            self.el1 = 18
            self.el2 = 21
        elif elevation_angles == "21_23":
            self.el1 = 21
            self.el2 = 24
        elif elevation_angles == "24_26":
            self.el1 = 24
            self.el2 = 27
        elif elevation_angles == "27_29":
            self.el1 = 27
            self.el2 = 30
        elif elevation_angles == "30_32":
            self.el1 = 30
            self.el2 = 33
        elif elevation_angles == "33_35":
            self.el1 = 33
            self.el2 = 36
        elif elevation_angles == "36_38":
            self.el1 = 36
            self.el2 = 39
        elif elevation_angles == "39_41":
            self.el1 = 39
            self.el2 = 42
        elif elevation_angles == "42_44":
            self.el1 = 42
            self.el2 = 45
        elif elevation_angles == "4_6":
            self.el1 = 0
            self.el2 = 11
        self.num_data = num_data
        self.n_steps = n_steps
        self.ZH_file_path_list = ZH_file_path_list
        self.ZDR_file_path_list = ZDR_file_path_list
        self.PHIDP_file_path_list = PHIDP_file_path_list
        self.RHOHV_file_path_list = RHOHV_file_path_list
        self.ZH_data = torch.zeros((40, self.n_steps, 301))
        self.ZDR_data = torch.zeros((40, self.n_steps, 301))
        self.PHIDP_data = torch.zeros((40, self.n_steps, 301))
        self.RHOHV_data = torch.zeros((40, self.n_steps, 301))
        self.savefolder = savefolder

    def data_load(self, index):
        #torch.cuda.synchronize()
        #loadbegin = time.time()
        #print(f"loaded {self.ZH_file_path_list[index]=}")
        rawdata_ZH = RAWData(self.ZH_file_path_list[index])
        #torch.cuda.synchronize()
        #loadZH = time.time()
        rawdata_ZDR = RAWData(self.ZDR_file_path_list[index])
        #torch.cuda.synchronize()
        #loadZDR = time.time()
        rawdata_PHIDP = RAWData(self.PHIDP_file_path_list[index])
        #torch.cuda.synchronize()
        #loadPHIDP = time.time()
        rawdata_RHOHV = RAWData(self.RHOHV_file_path_list[index])
        #torch.cuda.synchronize()
        #loadRHOHV = time.time()
        self.ZH_data = torch.tensor(rawdata_ZH.to_numpy()[0][self.el1:self.el2, :self.n_steps, :]).to(torch.float32)
        self.ZDR_data = torch.tensor(rawdata_ZDR.to_numpy()[0][self.el1:self.el2, :self.n_steps, :]).to(torch.float32)
        #print(f"{self.ZDR_data.shape=}")
        self.angle_offsets = rawdata_ZH.polar_blocks[0].az
        #print(f"{self.angle_offsets=}")
        self.PHIDP_data = torch.tensor(rawdata_PHIDP.to_numpy()[0][self.el1:self.el2, :self.n_steps, :]).to(torch.float32)
        self.RHOHV_data = torch.tensor(rawdata_RHOHV.to_numpy()[0][self.el1:self.el2, :self.n_steps, :]).to(torch.float32)
        #torch.cuda.synchronize()
        #loadend = time.time()
        self.preprocess()
        #torch.cuda.synchronize()
        #processend = time.time()
        #print(f"{loadend-loadbegin=}")
        #print(f"{loadZH-loadbegin=}")
        #print(f"{loadZDR-loadZH=}")
        #print(f"{loadPHIDP-loadZDR=}")
        #print(f"{loadRHOHV-loadPHIDP=}")
        #print(f"{processend-loadend=}")

    def preprocess(self):
        self.ZH_data = torch.where(torch.nan_to_num(self.ZH_data, nan=0)<0, 0, torch.nan_to_num(self.ZH_data, nan=0))
        #self.ZDR_data = torch.where(self.ZDR_data<0, -50, self.ZDR_data)
        #self.ZDR_data = torch.where(self.ZDR_data>8, -50, self.ZDR_data)
        self.ZDR_data = torch.nan_to_num(self.ZDR_data, nan=-50) # we will mask -50 later for loss computation
        self.PHIDP_data = torch.nan_to_num(self.PHIDP_data, nan=-50) # we will mask -50 later for loss computation
        self.RHOHV_data = torch.nan_to_num(self.RHOHV_data, nan=-50) # we will mask -50 later for loss computation
        self.ZDR_corrected = self.ZDR_data
        # normalize data here
        shortindex = torch.zeros_like(self.ZH_data).bool()
        longindex = torch.zeros_like(self.ZH_data).bool()
        shortindex[:, :118, 0] = True
        longindex[:, 118:, 0] = True
        # ZH: do nothing
        # Zdr: reduce mean of Zdr at 0 < ZH < 20 to 0.4
        to_subtract_all = []
        for el in range(self.el2-self.el1):
            Zhnow = self.ZH_data[el, :, :300] # [800, 300]
            Zdrnow = self.ZDR_data[el, :, :300]
            RHOHVnow = self.RHOHV_data[el, :, :300]
            cond1now = Zhnow < 20
            cond2now = Zhnow > 10
            cond3now = Zdrnow < 30
            cond4now = Zdrnow > -30 # eliminate nans
            #cond3now = 1
            #cond4now = 1
            cond5now = RHOHVnow > 0.95
            shortindex = torch.zeros_like(cond1now).bool()
            longindex = torch.zeros_like(cond1now).bool()
            shortindex[:118, :] = True
            longindex[118:, :] = True
            conds_short = cond1now*cond2now*cond3now*cond4now*cond5now*shortindex
            conds_long = cond1now*cond2now*cond3now*cond4now*cond5now*longindex
            meanshort = torch.mean(Zdrnow[conds_short])
            meanlong = torch.mean(Zdrnow[conds_long])
            #print(f"{meanshort=}")
            #print(f"{meanlong=}")
            
            to_subtract = torch.zeros(800)
            to_subtract[:118] = meanshort
            to_subtract[118:] = meanlong
            #print(f"{el=}, {meanshort=}")
            #print(f"{el=}, {meanlong=}")
            to_subtract_all.append(to_subtract)

        to_subtract_all = torch.stack(to_subtract_all, dim=0) # [40, 800]
        #torch.save(to_subtract_all, os.path.join(self.savefolder, "Zdr_subtract"))
        #print(f"{self.ZDR_data.shape=}")
        #print(f"{to_subtract_all.shape=}")
        self.ZDR_corrected = self.ZDR_data - to_subtract_all.unsqueeze(2).repeat(1, 1, 301)+0.3 # [40, 800] to [20, 800, 301]
        self.ZDR_corrected = torch.nan_to_num(self.ZDR_corrected, nan=-50) # meanshort could be nans
        #self.ZDR_corrected = torch.where(self.ZDR_corrected < -10, -50, self.ZDR_corrected)

    @property
    def duration(self):
        return self.dt * self.n_steps

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        try:
            self.data_load(idx)
            return [
                rearrange(self.ZH_data[:, :, :300], "a b c -> (a c) b"),
                rearrange(self.ZDR_corrected[:, :, :300], "a b c -> (a c) b"),
                rearrange(self.PHIDP_data[:, :, :300], "a b c -> (a c) b"),
                rearrange(self.RHOHV_data[:, :, :300], "a b c -> (a c) b"),
                self.angle_offsets
            ] # 5 deg
        except:
            print(f"load failed at {self.ZH_file_path_list[idx]=}")
            return [-100, -100, -100, -100, -100]
    
