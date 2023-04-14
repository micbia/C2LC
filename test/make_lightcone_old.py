import numpy as np, os, gc, sys
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
rank_proc = comm.Get_rank()
nprocs = comm.Get_size()

# Array job setup
rank_node = int(os.environ['SLURM_ARRAY_TASK_ID'])
ntasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

buffstart = comm.gather(True, root=0)
if(rank_proc == 0):
    if(all(buffstart)):
        print(' all processor on node=%d initialised successfully' %rank_node)
    print("___  ___      _          _     _       _     _                       \n|  \/  |     | |        | |   (_)     | |   | |                      \n| .  . | __ _| | _____  | |    _  __ _| |__ | |_ ___ ___  _ __   ___ \n| |\/| |/ _` | |/ / _ \ | |   | |/ _` | '_ \| __/ __/ _ \| '_ \ / _ \ \n| |  | | (_| |   <  __/ | |___| | (_| | | | | || (_| (_) | | | |  __/\n\_|  |_/\__,_|_|\_\___| \_____/_|\__, |_| |_|\__\___\___/|_| |_|\___|\n                                  __/ |                              \n                                 |___/                               ")

timer = Timer()
timer.start()

######################### INPUTS #########################
# Ushuu uses Planck 2015 cosmology parameters
cosmo = ExtendCosmology(cosmo)
path = '/work/backup/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'

#path_out = path+'MHI_Modi2019_ModelC/'
#dthet, dnu = 0.4 * u.deg, 390.*u.kHz      # for HIRAX
#z_min, z_max = 0.775, 2.55

path_out = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/test_MHI_Modi2019_ModelC/'
dthet, dnu = 0.5 * u.deg, 5.*u.MHz
z_min, z_max = 0.86, 2.46
z_space_dist = False
##########################################################

# define boxsize 
LB = 2000./cosmo.h * u.Mpc
nu0 = (1.42*u.GHz).to(u.MHz)
idx_z, cube_redshift = np.loadtxt('%sredshift.txt' %path, usecols=(0,1), unpack=True)

# Set loop starting index per processor and task
jobs_node = np.arange(0, idx_z.size)
range_node = jobs_node[jobs_node % ntasks == rank_node]
jobs_proc = np.arange(0, 100)
range_proc = jobs_proc[jobs_proc % nprocs == rank_proc]
print(' node=%d with rank=%d has tasks\n halodir : %s\n subvolume : %s' %(rank_node, rank_proc, str(range_node), str(range_proc)))

# calculate redshift bin
z_bin = get_z_output(zmin=z_min, zmax=z_max, fdepth_MHz=dnu)
lc_redshift = 0.5*(z_bin[1:] + z_bin[:-1])

# get observer position, supposed to be at z_min
robs = LB/2*np.array([1,1,0]) - cosmo.comoving_distance(z_min)*np.array([0,0,1])

# calculate angular bin
dr = np.deg2rad(dthet.value) * cosmo.comoving_transverse_distance(z_min)
dr_bin = np.arange(-LB.value/2, LB.value/2, dr.value)
mid_dr = 0.5*(dr_bin[1:] + dr_bin[:-1])

# calcualte volume
V_cell =  dr * dr* (cosmo.comoving_distance(z_bin[1:]) - cosmo.comoving_distance(z_bin[:-1]))

# save bins in file
np.savetxt('%sz_bin.txt' %path_out, np.array([z_bin[:-1], z_bin[1:], lc_redshift, cosmo.z2nu(lc_redshift), V_cell]).T, fmt='%.3f\t%.3f\t%.5f\t%.3f\t%.3f', header='z_l\tz_r\t\tz_mid\tnu [MHz]\tV_cell [Mpc^3]')
np.savetxt('%sdr_bin.txt' %path_out, np.array([dr_bin[:-1], dr_bin[1:], mid_dr]).T, fmt='%.5e', header='from observer coordinate system r_obs=%s\ndr_l\tdr_r\tdr_mid' %(str(robs)))

# compute the mHI model in the halo simulations
for i in range_node:
    i_z0, z0 = idx_z[i], cube_redshift[i]
    fname = '%slc_z%d_%.1f%s_%d%s' %(path_out, i_z0, dthet.value, dthet.unit, dnu.value, dnu.unit)
    
    timer.lap('rank=%d start halodir%d' %(rank_proc, i))
    if not (os.path.exists(fname)):
        if(rank_proc == 0):
            lc_i = np.zeros((lc_redshift.size, mid_dr.size, mid_dr.size))

        for j in range_proc:
            # get halo variables for redshift z file (mvir, rvir, rx, ry, rz, vx, vy, vz)
            mvir, rvir, rx, ry, rz, _, _, v_per = read_cbin('%shalodir_%03d/halolist_z%.2f_%d.bin' %(path, i_z0, z0, j), dimensions=2)
            
            if('Padmanabhan' in path_out):
                mean_overd = mvir/cosmo.h / (4./3*np.pi*rvir/1000./cosmo.h)**3 / cosmo.critical_density(z0).to(u.Msun/u.Mpc**3).value
                del rvir
                gc.collect()
            
            first = True
            for i_z, z in enumerate(lc_redshift):
                # get HI mass in halos
                if('Padmanabhan' in path_out):
                    mHI = cosmo.MHI_Padmanabhan2017(mH=mvir/cosmo.h, z=z, delta_c=mean_overd)
                elif('Modi2019_ModelA' in path_out):
                    mHI = cosmo.MHI_Modi2019(Mh=mvir/cosmo.h, z=z, model='A')
                elif('Modi2019_ModelC' in path_out):
                    mHI = cosmo.MHI_Modi2019(Mh=mvir/cosmo.h, z=z, model='C')

                # get only halos that contain hydrogen
                mask_HI = mHI != 0

                # calculate the LOS distance (or redshit) for the halos
                if(z_space_dist):
                    z_obs = (1+cosmo.cdist2z(rz / cosmo.h - robs[2].value)) * (1+vz/cst.c.to(u.km/u.s).value)-1
                    mask_z = (z_obs >= z_bin[i_z]) * (z_obs < z_bin[i_z+1])
                else:
                    rz_obs = rz[mask_HI] / cosmo.h - robs[2].value
                    mask_z = (rz_obs >= cosmo.comoving_distance(z_bin[i_z]).value) * (rz_obs < cosmo.comoving_distance(z_bin[i_z+1]).value)

                # get halo position within the redshift bin value
                rh = np.vstack((rx[mask_HI][mask_z], ry[mask_HI][mask_z])).T / cosmo.h

                # get halos xy-position from observer POV
                rh_obs = rh - robs[:2].value

                # get HI mass in redshift bin
                mHI = mHI[mask_HI][mask_z]

                ret = stats.binned_statistic_2d(rh_obs[:,0], rh_obs[:,1], mHI, 'sum', bins=[dr_bin, dr_bin])

                if(first and i_z == 0):
                    lc_slice = ret.statistic #(ret.statistic * u.Msun / V_cell).cgs.value
                    first = False
                else:
                    lc_slice += ret.statistic

                recvbuf_lc = comm.gather(lc_slice, root=0)
                recvbuf_iz = comm.gather(i_z, root=0)

                if(rank_proc == 0):
                    # gather processor results to rank=0
                    for i_buff, iz_proc in enumerate(recvbuf_iz):
                        if(i_buff == 0):
                            lc_i[iz_proc] = recvbuf_lc[i_buff]
                        else:
                            lc_i[iz_proc] += recvbuf_lc[i_buff]
                    
                    # save cube of mHI
                    np.save(fname, lc_i)
    else:
        print(' %s exist already - skip' %fname)
        lc_i = np.load(fname)
    
    bufflc = comm.gather(True, root=0)
    if(rank_proc == 0):
        if not (all(bufflc)):
            raise Exception(' Some processor failed the redshift part - Stopping')
            sys.exist()

        # calculate the fraction of neutral HI
        xHI = (lc_i/V_cell[..., np.newaxis, np.newaxis] / (cosmo.mean_molecular * cst.m_p * cosmo.nH0)).cgs.value

        # calculate differential brightness
        dT = cosmo.dTb(xHI, z=z0)
        np.save('%sdT_z%.2f_%sdeg_%d%s.npy' %(path_out, z0, str(dthet.value), dthet.unit, dnu, dnu.unit), dT.value) # in mK

buffmake = comm.gather(True, root=0)
# Make the lightcone, one slice at a time
if(rank_node == 0 and rank_proc == 0):
    if not (all(buffmake)):
        raise Exception(' Some processor failed the halodir part - Stopping')
        sys.exist()
    lightcone = np.zers_like(lc_i)
    for i_l in tqdm(range(lc_redshift.size)):
        z = lc_redshift[i_l]

        i_cube = np.digitize(z, cube_redshift, right=True)
        z_low, z_high = cube_redshift[i_cube-1], cube_redshift[i_cube]

        data_low = np.load('%sdT_z%.2f_%sdeg_%dMHz.npy' %(path_out, z_low, str(dthet.value), dnu), mmap_mode='r')
        data_high = np.load('%sdT_z%.2f_%sdeg_%dMHz.npy' %(path_out, z_high, str(dthet.value), dnu), mmap_mode='r')

        slice_low = data_low[i_l]
        slice_high = data_high[i_l]

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

        lightcone[i_l] = slice_interp

    # save lightcone
    np.save(fout_lc, lightcone)

    i_z, i_lc = lightcone.shape[0]//2, lightcone.shape[1]//2

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])

    ax0 = fig.add_subplot(gs[0,0])
    im = ax0.pcolormesh(lc_redshift, angl, lightcone[:,i_lc,:].T, cmap='jet', norm=LogNorm())
    ax0.vlines(x=lc_redshift[i_z], ymin=angl.min(), ymax=angl.max(), colors='lime', ls='--')
    ax0.set_xlabel('z'), ax0.set_ylabel(r'$\theta$ [deg]')
    ax0.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax0.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title('z = %.3f' %lc_redshift[i_z])
    im = ax1.pcolormesh(angl, angl, lightcone[i_z], cmap='jet')#, norm=LogNorm())
    ax1.set_xlabel(r'$\theta$ [deg]'), ax1.set_ylabel(r'$\theta$ [deg]')
    plt.colorbar(im, ax=ax1, cax=fig.add_axes([0.91, 0.15, 0.01, 0.7]), label=r'$\delta T_b$ [mK]')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    plt.savefig('%slc_dT_%s_%sdeg_%dMHz.png' %(path_out, typ_model, str(dthet), dnu), bbox_inches='tight')
    plt.clf()

timer.stop()
print(' rank=%d on node=%d is done' %(rank_node, rank_proc))