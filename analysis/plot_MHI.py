import numpy as np, h5py, sys, matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from astropy.cosmology import Planck15 as cosmo

sys.path.insert(0,'../')
from utils.utils_cosm import ExtendCosmology, MHI_Modi2019
from utils.other_utils import PercentContours

cosmo = ExtendCosmology(cosmo)
path = '/home/mbianco/codes/hirax/output_sim/'

redshift = np.array([0.8, 1.2, 2.0])
mHalo = np.logspace(7, 14, 1000) # in Msun

for z in redshift:
    mHI = MHI_Modi2019(Mh=mHalo, z=z, cosmo=cosmo, model='A')
    mHI_Pad = cosmo.MHI_Padmanabhan2017(Mh=mHalo, z=z, delta_c=200.)
    plt.loglog(mHalo, mHI, label='z=%.3f' %z)
    plt.loglog(mHalo, mHI_Pad, label='z=%.3f' %z)
plt.ylim(1e-4, 1e13)
plt.xlabel(r'$M_{halo}$ [Msun]'), plt.ylabel(r'$M_{HI}$ [Msun]')
plt.legend()
plt.savefig('%smodelA_Modi2019_mHI.png' %(path), bbox_inches='tight'), plt.clf()

# ----------------------------------------------------------------------------
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = (15,8)
path_in = '/work/backup/ska/HIRAX/CAMELS_dataset/MHI_grids/'

z = 1.496 #0.997
redshift = [0.997, 1.496, 2.002]
box_size, mesh_size = [75, 205], [1820, 2500]

for z in redshift:
    for i, (LB, NP) in enumerate(zip(box_size, mesh_size)):
        f = h5py.File('%sM_HI_%d_%d_z=%.3f.hdf5' %(path_in, LB, NP, z), 'r')
        mHalo_sim = f['Mass'][:] / cosmo.h
        mHI_sim = f['M_HI'][:] / cosmo.h
        f.close()
        
        min_M, max_M = mHalo_sim.min(), mHalo_sim.max()
        print(' Simulation %d Mpc/h: M_halo(z=%.3f) = [%.3e, %.3e]' %(LB, z, min_M, max_M))

        mHalo_model = np.logspace(np.log10(min_M), np.log10(max_M), 1000)
        mHI_modelA = cosmo.MHI_Modi2019(Mh=mHalo_model, z=z, model='A')
        mHI_modelC = cosmo.MHI_Modi2019(Mh=mHalo_model, z=z, model='C')
        mHI_Pad = cosmo.MHI_Padmanabhan2017(Mh=mHalo_model, z=z, delta_c=180.)

        plt.title('z=%.3f' %z)
        PercentContours(x=mHalo_sim, y=mHI_sim, bins='log', colour='lime', style=['-', '--'], perc_arr=[0.99, 0.95])
        plt.scatter(x=mHalo_sim, y=mHI_sim, s=10, color='tab:blue', label=r'%d Mpc/h $%d^3$ particles' %(LB, NP), marker='.', alpha=0.8)
        plt.loglog(mHalo_model, mHI_modelA, label='Model A: Modi+ (2019)', color='tab:orange', ls='-')
        plt.loglog(mHalo_model, mHI_modelC, label='Model C: Modi+ (2019)', color='tab:orange', ls='--')
        plt.loglog(mHalo_model, mHI_Pad, label=r' $\Delta_c=180$, Padmanabhan+ (2017)', color='tab:red', ls='-')
        plt.xlabel(r'$M_{halo}$ $[M_{\odot}]$'), plt.ylabel(r'$M_{HI}$ $[M_{\odot}]$')
        
        plt.ylim(1e-4, 1e12), plt.xlim()
        plt.legend(loc=4)
        plt.savefig('%scompare_MHI_%dMpc_z%.3f.png' %(path, LB, z), bbox_inches='tight'), plt.clf()
