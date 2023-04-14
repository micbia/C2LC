import numpy as np, os, gc, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from matplotlib.colors import LogNorm
from tqdm import tqdm
from scipy import stats
from mpi4py import MPI

import astropy.units as u
import astropy.constants as cst
from astropy.cosmology import Planck15 as cosmo

from utils.other_utils import read_cbin, Timer
from utils.utils_cosm import ExtendCosmology, get_z_output

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

buffstart = comm.gather(True, root=0)
if(rank == 0):
    if(all(buffstart)):
        print(' all processor initialised successfully')
    print("___  ___      _          _     _       _     _                       \n|  \/  |     | |        | |   (_)     | |   | |                      \n| .  . | __ _| | _____  | |    _  __ _| |__ | |_ ___ ___  _ __   ___ \n| |\/| |/ _` | |/ / _ \ | |   | |/ _` | '_ \| __/ __/ _ \| '_ \ / _ \ \n| |  | | (_| |   <  __/ | |___| | (_| | | | | || (_| (_) | | | |  __/\n\_|  |_/\__,_|_|\_\___| \_____/_|\__, |_| |_|\__\___\___/|_| |_|\___|\n                                  __/ |                              \n                                 |___/                               ")

timer = Timer()
timer.start()

######################### INPUTS #########################
# Ushuu uses Planck 2015 cosmology parameters
cosmo = ExtendCosmology(cosmo)
path = '/work/backup/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'

#path_out = path+'MHI_Modi2019_ModelA/'
#path_out = path+'MHI_Modi2019_ModelA_rsd/'
#path_out = path+'MHI_Modi2019_ModelA_M10cut/'
#path_out = path+'MHI_Modi2019_ModelC/'
#path_out = path+'MHI_Modi2019_ModelC_rsd/'
path_out = path+'MHI_Modi2019_ModelC_M10cut/'
#path_out = path+'MHI_Padmanabhan/'
#path_out = path+'MHI_Padmanabhan_rsd/'
#path_out = path+'MHI_Padmanabhan_M10cut/'

dthet, dnu = 0.4 * u.deg, 390.*u.kHz      # for HIRAX
z_min, z_max = 0.8, 2.37 #0.775, 2.55

#path_out = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/test_MHI_Modi2019_ModelC/'
#dthet, dnu = 0.5 * u.deg, 5.*u.MHz
#z_min, z_max = 0.86, 2.46

interpolation = 'sigmoid'
##########################################################
if('rsd' in path_out):
    z_space_dist = True
else:
    z_space_dist = False

# define boxsize 
LB = 2000./cosmo.h * u.Mpc
idx_z, cube_redshift = np.loadtxt('%sredshift.txt' %path, usecols=(0,1), unpack=True)
if(rank == 0):
    os.system('cp %sredshift.txt %sredshift.txt' %(path, path_out))

# Set loop starting index per processor and task
jobs_proc = np.arange(0, cube_redshift.size)
range_proc = jobs_proc[jobs_proc % nprocs == rank]
print(' rank=%d has tasks\n halodir : %s' %(rank, str(range_proc)))

# calculate redshift bin
z_bin = get_z_output(zmin=z_min, zmax=z_max, fdepth_MHz=dnu)
lc_redshift = 0.5*(z_bin[1:] + z_bin[:-1])
los_bin = cosmo.comoving_distance(z_bin)
mid_los = 0.5*(los_bin[1:] + los_bin[:-1])

# get observer position, supposed to be at z_min
robs = LB/2*np.array([1,1,0]) - cosmo.comoving_distance(z_min)*np.array([0,0,1])

# calculate angular bin
dr = np.deg2rad(dthet.value) * cosmo.comoving_transverse_distance(z_min)
dr_bin = np.arange(-LB.value/2, LB.value/2, dr.value)
mid_dr = 0.5*(dr_bin[1:] + dr_bin[:-1])
angl = cosmo.cdist2deg(dr_bin*u.Mpc, lc_redshift.min()).value

# calcualte volume
V_cell =  dr * dr* (cosmo.comoving_distance(z_bin[1:]) - cosmo.comoving_distance(z_bin[:-1]))

# save bins in file
if(rank == 0):
    np.savetxt('%sz_bin.txt' %path_out, np.array([z_bin[:-1], z_bin[1:], lc_redshift, cosmo.z2nu(lc_redshift), V_cell.value]).T, fmt='%.3f\t%.3f\t%.5f\t\t%.3f\t%.3f', header='z_l\tz_r\tz_mid\t\tnu [MHz]\tV_cell [Mpc^3]')
    np.savetxt('%sdr_bin.txt' %path_out, np.array([dr_bin[:-1], dr_bin[1:], mid_dr]).T, fmt='%.5e', header='from observer coordinate system r_obs=%s\ndr_l\tdr_r\tdr_mid' %(str(robs)))

# compute the mHI model in the halo simulations
for i in range_proc:
    i_z0, z0 = idx_z[i], cube_redshift[i]
    fname = '%slc_z%d_%.1f%s_%d%s' %(path_out, i_z0, dthet.value, dthet.unit, dnu.value, dnu.unit)
    
    timer.lap('rank=%d start halodir%d' %(rank, i))
    if not (os.path.exists(fname)):
        lc_i = np.zeros((lc_redshift.size, mid_dr.size, mid_dr.size))

        for j in range(0, 100):
            # get halo variables for redshift z file (mvir, rvir, rx, ry, rz, vx, vy, vz)
            mvir, rvir, rx, ry, rz, _, _, vz = read_cbin('%shalodir_%03d/halolist_z%.2f_%d.bin' %(path, i_z0, z0, j), dimensions=2)

            # get HI mass in halos
            if('Padmanabhan' in path_out):
                mean_overd = mvir/cosmo.h / (4./3*np.pi*rvir/cosmo.h/1000.)**3 / cosmo.critical_density(z0).to(u.Msun/u.Mpc**3).value
                del rvir
                gc.collect()
                mHI = cosmo.MHI_Padmanabhan2017(Mh=mvir/cosmo.h, z=z0, delta_c=mean_overd)
            elif('Modi2019_ModelA' in path_out):
                mHI = cosmo.MHI_Modi2019(Mh=mvir/cosmo.h, z=z0, model='A')
            elif('Modi2019_ModelC' in path_out):
                mHI = cosmo.MHI_Modi2019(Mh=mvir/cosmo.h, z=z0, model='C')
            
            if('cut' in path_out):
                cut = int(path_out[path_out.rfind('_M')+2:path_out.rfind('cut')])
                mHI[mvir/cosmo.h <= np.power(10, cut)] = 0.

            # get only halos that contain hydrogen
            mask_HI = mHI != 0

            # get halos xy-position from observer POV
            rx_obs = rx[mask_HI] / cosmo.h - robs[0].value
            ry_obs = ry[mask_HI] / cosmo.h - robs[1].value

            # get halos that have non-zero HI mass
            mHI = mHI[mask_HI]
            del rx, ry
            gc.collect()

            # calculate the LOS distance (or redshit) for the halos
            if(z_space_dist):
                z_obs = (1+cosmo.cdist2z(rz[mask_HI] / cosmo.h - robs[2].value)) * (1+vz[mask_HI]/cst.c.to(u.km/u.s).value) - 1
                ret = stats.binned_statistic_dd([rx_obs, ry_obs, z_obs], mHI, 'sum', bins=[dr_bin, dr_bin, z_bin])
            else:
                rz_obs = rz[mask_HI] / cosmo.h - robs[2].value
                ret = stats.binned_statistic_dd([rx_obs, ry_obs, rz_obs], mHI, 'sum', bins=[dr_bin, dr_bin, los_bin])
            del vz
            gc.collect()

            if(j == 0):
                lc_i = ret.statistic
            else:
                lc_i += ret.statistic

            # save cube of mHI
            np.save(fname, lc_i)
    else:
        print(' %s exist already - skip' %fname)
        lc_i = np.load(fname)
    
    # calculate the fraction of neutral HI
    xHI = (lc_i * u.Msun / (V_cell * cosmo.mean_molecular * cst.m_p * cosmo.nH0)).cgs.value

    # calculate differential brightness
    dT = cosmo.dTb(xHI, z=z0)

    np.save('%sdT_z%.2f_%.1f%s_%d%s.npy' %(path_out, z0, dthet.value, dthet.unit, dnu.value, dnu.unit), dT.value) # in mK


buffmake = comm.gather(True, root=0)
if(rank == 0):
    print('\n Interpolate cubes to create the lightcone:\t z_space_dist=%s' %(str(z_space_dist)))
    cube_redshift = np.sort(cube_redshift)

    if not (all(buffmake)):
        raise Exception(' Some processor failed the halodir part - Stopping')
        sys.exist()

    typ_model = path_out[path_out[:-1].rfind('/')+1:-1]
    fout_lc = '%slc_dT_%s_%.1f%s_%d%s.npy' %(path_out, typ_model, dthet.value, dthet.unit, dnu.value, dnu.unit)

    if not (os.path.exists(fout_lc)):
        lightcone = np.zeros_like(lc_i)
        for i_l in tqdm(range(lc_redshift.size)):
            z = lc_redshift[i_l]

            i_cube = np.digitize(z, cube_redshift, right=True)
            z_low, z_high = cube_redshift[i_cube-1], cube_redshift[i_cube]

            data_low = np.load('%sdT_z%.2f_%.1f%s_%d%s.npy' %(path_out, z_low, dthet.value, dthet.unit, dnu.value, dnu.unit), mmap_mode='r')
            data_high = np.load('%sdT_z%.2f_%.1f%s_%d%s.npy' %(path_out, z_high, dthet.value, dthet.unit, dnu.value, dnu.unit), mmap_mode='r')

            slice_low = data_low[...,i_l]
            slice_high = data_high[...,i_l]

            if(interpolation == 'linear'):
                slice_interp = ((z-z_low)*slice_high + (z_high - z)*slice_low)/(z_high-z_low)
            elif(interpolation == 'cubic'):
                raise ValueError(' Method not implemented: %s' %interpolation)
            elif(interpolation == 'sigmoid'):
                zp = (z-z_low)/(z_high-z_low) * 20. - 10.   # normalise betwen -10 and 10
                w = 1./(1.+np.exp(-2.*zp))                  # redshift weight
                slice_interp = slice_low*(1. - w) + slice_high * w
            else:
                raise Exception('Unknown interpolation method: %s' %interpolation)

            lightcone[...,i_l] = slice_interp

        # save lightcone
        np.save(fout_lc, lightcone)
    else:
        lightcone = np.load(fout_lc)
    
    i_z, i_lc = lightcone.shape[2]//2, lightcone.shape[1]//2

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])

    ax0 = fig.add_subplot(gs[0,0])
    im = ax0.pcolormesh(lc_redshift, angl, lightcone[:,i_lc,:], cmap='jet')#, norm=LogNorm())
    ax0.vlines(x=lc_redshift[i_z], ymin=angl.min(), ymax=angl.max(), colors='lime', ls='--')
    ax0.set_xlabel('z'), ax0.set_ylabel(r'$\theta$ [deg]')
    ax0.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax0.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title('z = %.3f' %lc_redshift[i_z])
    im = ax1.pcolormesh(angl, angl, lightcone[...,i_z], cmap='jet')#, norm=LogNorm())
    ax1.set_xlabel(r'$\theta$ [deg]'), ax1.set_ylabel(r'$\theta$ [deg]')
    plt.colorbar(im, ax=ax1, cax=fig.add_axes([0.91, 0.15, 0.01, 0.7]), label=r'$\delta T_b$ [mK]')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    plt.savefig('%slc_dT_%s_%.1f%s_%d%s.png' %(path_out, typ_model, dthet.value, dthet.unit, dnu.value, dnu.unit), bbox_inches='tight')
    plt.clf()

timer.stop()
print(' rank=%d is done' %(rank))