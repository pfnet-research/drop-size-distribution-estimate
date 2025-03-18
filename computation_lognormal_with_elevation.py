#!/usr/bin/env python

"""
This demo replicates Fig. 7.7 from Bringi and Chandrasekar (2001),
Polarimetric Weather Radar: Principles and Applications. It shows the
specific differential phase (Kdp) normalized by the total water content (W)
as a function of the mass-weighted mean diameter.
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy import constants
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar, tmatrix_aux

def axis_ratio(D):
    # the Beard-Chuang axis ratio (eq. 7.3)
    return 1.0/(1.0048 + 5.7e-4*D - 2.628e-2*D**2 + 3.682e-3*D**3 -
                1.677e-4*D**4)
    #return 1.0/(0.9951+0.0251*D-0.03644*D*D+0.00503*D*D*D-0.0002492*D*D*D*D)

# initialize a scatterer object
scatterer = Scatterer()
scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)

for i, mean in enumerate(np.linspace(0, 6, 13)):
    print(f"{mean=}")
    # set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
    scatterer.or_pdf = orientation.gaussian_pdf(std=10.0, mean=mean)  # orientation PDF
    scatterer.orient = orientation.orient_averaged_fixed  # averaging method
    
    # set up PSD integration
    scatterer.psd_integrator = psd.PSDIntegrator()
    scatterer.psd_integrator.D_max = 8.0  # maximum diameter considered
    scatterer.psd_integrator.geometries = (tmatrix_aux.geom_horiz_forw,)
    scatterer.psd_integrator.axis_ratio_func = axis_ratio

    #Dm = np.linspace(0.5, 3.8, 1000)  # range of Dm (mm)
    #lam = 4.0/Dm  # corresponding lambda parameters
    #mus = np.linspace(2.5, 12.5, 100)
    
    #W = np.pi*1e3*(Dm/4.0)**4  # corresponding water content

    wavelengths = constants.c/np.array([9.425e9]) * 1e3  # in mm
    ref_indices = [complex(7.849, 2.388)]
    labels = ["10 GHz"]
    styles = ["-"]

    # this calculates Kdp for the given lambda parameter
    def get_Kdp(loc, log_scale_correction):
        #lambd = 1.935+0.735*mu+0.0365*mu*mu
        #lambd = -1.2750+1.4286*mu-0.0799*mu*mu
        #scatterer.psd = ExponentialPSD(lam=lam)  # set exponential PSD
        #scatterer.psd = psd.GammaPSD(Nw=1, mu=mu, D0=D0)
        log_scale = log_scale_correction-0.8729+0.0291*loc-0.0873*loc**2-0.0442*loc**3-0.0925*loc**4
        scatterer.psd = psd.LogNormalPSD(Nw=1, loc=loc, log_scale=log_scale, D_max=10.0)
        return radar.Kdp(scatterer)

    def get_Zdr(loc, log_scale_correction):
        #lambd = 1.935+0.735*mu+0.0365*mu*mu
        #lambd = -2.6744+1.2012*mu+0.0318*mu*mu
        #D0 = (mu+3.67)/lambd
        #scatterer.psd = ExponentialPSD(lam=lam)  # set exponential PSD
        #scatterer.psd = psd.GammaPSD(Nw=1, mu=mu, D0=D0)
        log_scale = log_scale_correction-0.8729+0.0291*loc-0.0873*loc**2-0.0442*loc**3-0.0925*loc**4
        scatterer.psd = psd.LogNormalPSD(Nw=1, loc=loc, log_scale=log_scale, D_max=10.0)
        if radar.refl(scatterer) == 0:
            return 10
        else:
            #print(f"{D0=}, {10*np.log10(radar.Zdr(scatterer))=}")
            return 10*np.log10(radar.Zdr(scatterer))

    def get_ZH(loc, log_scale_correction):
        #lambd = 1.935+0.735*mu+0.0365*mu*mu
        #lambd = -2.6744+1.2012*mu+0.0318*mu*mu
        #D0 = (mu+3.67)/lambd
        #scatterer.psd = ExponentialPSD(lam=lam)  # set exponential PSD
        #scatterer.psd = psd.GammaPSD(Nw=1, mu=mu, D0=D0)
        log_scale = log_scale_correction-0.8729+0.0291*loc-0.0873*loc**2-0.0442*loc**3-0.0925*loc**4
        scatterer.psd = psd.LogNormalPSD(Nw=1, loc=loc, log_scale=log_scale, D_max=10.0)
        if radar.refl(scatterer) == 0:
            return 100
        else:
            return 10*np.log10(radar.refl(scatterer))

    def get_A(loc, log_scale_correction):
        #scatterer.psd = ExponentialPSD(lam=lam)  # set exponential PSD
        #lambd = 1.935+0.735*mu+0.0365*mu*mu
        #lambd = -2.6744+1.2012*mu+0.0318*mu*mu
        #D0 = (mu+3.67)/lambd
        #scatterer.psd = psd.GammaPSD(Nw=1, mu=mu, D0=D0)
        log_scale = log_scale_correction-0.8729+0.0291*loc-0.0873*loc**2-0.0442*loc**3-0.0925*loc**4
        scatterer.psd = psd.LogNormalPSD(Nw=1, loc=loc, log_scale=log_scale, D_max=10.0)
        return [radar.Ai(scatterer, h_pol=True), radar.Ai(scatterer, h_pol=False)]

    fig_kdp = plt.figure()
    fig_Zdr = plt.figure()
    fig_ZH = plt.figure()
    fig_A = plt.figure()

    ax_kdp = fig_kdp.add_subplot(111)
    ax_Zdr = fig_Zdr.add_subplot(111)
    ax_ZH = fig_ZH.add_subplot(111)
    ax_A = fig_A.add_subplot(111)
    loc_list = np.linspace(-1.2, 1.2, 61)
    log_scale_correction_list = np.linspace(-0.2, 0.2, 51)
    #mulist = np.linspace(1.0, 10.0, 91)
    #lambd_list = -1.2750+1.4286*mulist+0.0799*mulist*mulist
    #lambd_correction = np.linspace(-1.0, 1.0, 51)
    kdp_array = np.empty((61, 51))
    zdr_array = np.empty((61, 51))
    zh_array = np.empty((61, 51))
    ah_array = np.empty((61, 51))
    av_array = np.empty((61, 51))
    #for (wl, m, label, style) in zip(wavelengths, ref_indices, labels, styles):
    scatterer.wavelength = wavelengths[0]
    scatterer.m = ref_indices[0]
    print(f"{scatterer.wavelength=}")
    print(f"{scatterer.m=}")
    # initialize lookup table
    scatterer.psd_integrator.init_scatter_table(scatterer)        
    for j in range(61):
        for k in range(51):
            kdp_array[j, k] = get_Kdp(loc_list[j], log_scale_correction_list[k])
            zdr_array[j, k] = get_Zdr(loc_list[j], log_scale_correction_list[k])
            zh_array[j, k] = get_ZH(loc_list[j], log_scale_correction_list[k])
            ah_array[j, k] = get_A(loc_list[j], log_scale_correction_list[k])[0]
            av_array[j, k] = get_A(loc_list[j], log_scale_correction_list[k])[1]

    np.savetxt(f"lognorm_kdp_el{i}.txt", kdp_array)
    np.savetxt(f"lognorm_Zdr_el{i}.txt", zdr_array)
    np.savetxt(f"lognorm_ZH_el{i}.txt", zh_array)
    np.savetxt(f"lognorm_AH_el{i}.txt", ah_array)
    np.savetxt(f"lognorm_AV_el{i}.txt", av_array)


    '''
    for (wl, m, label, style) in zip(wavelengths, ref_indices, labels, styles):
    scatterer.wavelength = wl
    scatterer.m = m
    # initialize lookup table
    scatterer.psd_integrator.init_scatter_table(scatterer)
    Kdp = np.array([get_Kdp(l) for l in mus])
    Zdr = np.array([get_Zdr(l) for l in mus])
    ZH = np.array([get_ZH(l) for l in mus])
    Ah = np.array([get_A(l)[0] for l in mus])
    Av = np.array([get_A(l)[1] for l in mus])
    ax_kdp.plot(mus, 1e6*Kdp, ls=style, label=label)  # 1e6 for unit conversion
    ax_Zdr.plot(mus, Zdr, ls=style, label=label)  #
    ax_ZH.plot(mus, ZH, ls=style, label=label)  #
    ax_A.plot(mus, Ah, ls=style, label=label+" Ah")  # 1e6 for unit conversion
    ax_A.plot(mus, Av, ls=style, label=label+" AV")  # 1e6 for unit conversion
    
    np.savetxt(f"mu_AH_el{i}.txt", Ah)
    np.savetxt(f"mu_AV_el{i}.txt", Av)
    np.savetxt(f"mu_kdp_el{i}.txt", Kdp)
    np.savetxt(f"mu_Zdr_el{i}.txt", Zdr)
    np.savetxt(f"mu_ZH_el{i}.txt", ZH)
    '''
    
    '''
    ax_kdp.set_xlabel(r"$\mu$")
    ax_kdp.set_ylabel(r"$K_{dp}$ $\mathrm{(\degree \, km^{-1} / g \, m^{-3})}$")
    ax_kdp.legend(loc='best')
    fig_kdp.savefig(f"Kdp_result_el{i}")

    ax_Zdr.set_xlabel(r"$\mu$")
    ax_Zdr.set_ylabel(r"$Z_{dr}$ $\mathrm{(dB)}$")
    ax_Zdr.legend(loc='best')
    fig_Zdr.savefig(f"Zdr_result_el{i}")

    ax_ZH.set_xlabel(r"$\mu$")
    ax_ZH.set_ylabel(r"$Z_{H}$ $\mathrm{(dB \, / g \, m^{-3})}$")
    ax_ZH.legend(loc='best')
    fig_ZH.savefig(f"ZH_result_el{i}")

    ax_A.set_xlabel(r"$\mu$")
    ax_A.set_ylabel(r"$A$ $\mathrm{(dB \, km^{-1}/ g \, m^{-3})}$")
    ax_A.legend(loc='best')
    fig_A.savefig(f"A_result_el{i}")
    '''
    
    print(f"finished elevation {i}")
