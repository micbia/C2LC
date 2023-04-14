import numpy as np, sys, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

import astropy.units as u
import astropy.constants as cst
from astropy.cosmology import Planck15 as cosmo

sys.path.insert(0,'../')
from utils.utils_cosm import ExtendCosmology

cosmo = ExtendCosmology(cosmo)

#path = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/MHI_Padmanabhan/'
path = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/MHI_Modi2019_ModelC/'

loop_range = range(21, 37)

redshift, Vcell = np.loadtxt('%sz_bin.txt' %(path), usecols=(2, 4), unpack=True)
Vcell *= u.Mpc**3
angl = cosmo.cdist2deg(np.loadtxt('%sdr_bin.txt' %(path), usecols=(2))*u.Mpc, redshift.min())
FoV = (angl.max()-angl.min())
print('z = [%.3f, %.3f]\t FoV = %.2f %s' %(redshift.min(), redshift.max(), FoV.value, FoV.unit))
angl = angl.value

for i_z0 in loop_range:
    z = redshift[i_z0]
    fname = '%slc_z%d_0.4deg_390kHz.npy' %(path, i_z0)

    # get HI mass
    mHI = np.load(fname) * u.Msun

    # calculate the fraction of neutral HI
    #xHI = (mHI/Vcell[..., np.newaxis, np.newaxis] / cosmo.mean_molecular / cst.m_p / cosmo.nH0).cgs.value
    xHI = (mHI/Vcell[np.newaxis, np.newaxis, ...] / cosmo.mean_molecular / cst.m_p / cosmo.nH0).cgs.value

    # calculate differential brightness
    dT = cosmo.dTb(xHI, z=z)
    np.save('%sdT_z%.2f_0.5deg_5MHz.npy' %(path, z), dT.value) # in mK

    #print('mesh =', mHI.shape)
    i_z, i_lc = mHI.shape[2]//2, mHI.shape[1]//2
    #i_z, i_lc = 1000, mHI.shape[1]//2

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])

    ax0 = fig.add_subplot(gs[0,0])
    ax0.pcolormesh(redshift, angl, dT[:,i_lc,:].value, cmap='jet')#, norm=LogNorm())
    #ax0.pcolormesh(redshift, angl, mHI[:,i_lc,:].value, cmap='jet', norm=LogNorm())
    ax0.vlines(x=redshift[i_z], ymin=angl.min(), ymax=angl.max(), colors='lime', ls='--')
    ax0.set_xlabel('z'), ax0.set_ylabel(r'$\theta$ [deg]')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title('z = %.3f' %z)
    im = ax1.pcolormesh(angl, angl, dT[...,i_z].value, cmap='jet')#, norm=LogNorm())
    #im = ax1.pcolormesh(angl, angl, mHI[...,i_z].value, cmap='jet', norm=LogNorm())
    ax1.set_xlabel(r'$\theta$ [deg]'), ax1.set_ylabel(r'$\theta$ [deg]')

    plt.colorbar(im, ax=ax1, cax=fig.add_axes([0.91, 0.15, 0.01, 0.7]), label=r'$\delta T_b$ [%s]' %(dT.unit))
    plt.savefig('%slc_slice_z%d.png' %(path, i_z0), bbox_inches='tight')
    plt.clf()
