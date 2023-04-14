import numpy as np, os
import astropy.units as u
import astropy.constants as cst

from tqdm import tqdm
from scipy import stats
from astropy.cosmology import Planck15 as cosmo
from utils.read_halo import read_cbin
from utils.utils_cosm import z2nu, MHI_Modi2019

from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

path = '/work/backup/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'
path_out = path+'MHI_Modi2019_ModelA/'
dthet, dnu = 0.5 * u.deg, 5.*u.MHz
nu0 = (1.42*u.GHz).to(u.MHz)

# h = 0.6774
# Mvir in Msun/h
# Rvir in comoving kpc/h
# rx, ry and rz in comoving Mpc/h
# vx, vy, and vz in physical km/s (peculiar velocity)

# define boxsize and first redshift
LB = 2000./cosmo.h * u.Mpc
idx_z, redshift = np.loadtxt('%sredshift.txt' %path, usecols=(0,1), unpack=True)

# Set loop starting index per processor
loop_start, loop_end = 0, idx_z.size+1
#loop_start, loop_end = 0, 11
perrank = (loop_end-loop_start)//nprocs

i_start = int(loop_start+rank*perrank)
if(rank != nprocs-1):
    i_end = int(loop_start+(rank+1)*perrank)
else:
    i_end = loop_end
print(' rank=%d has tasks from %d to %d' %(rank, i_start, i_end))

# calculate redshift bin
z0 = redshift.min()
zi, zi1 = z0, 0
z_bin = [zi]
while zi1 <= redshift.max():
    zi1 = ((zi + dnu/nu0 * (1+zi)) / (1-dnu/nu0*(1+zi))).value
    z_bin.append(zi1)
    #print(zi, zi1, np.round(z2nu(zi) - z2nu(zi1)))
    zi = zi1

z_bin = np.array(z_bin)
mid_z = 0.5*(z_bin[1:] + z_bin[:-1])

# get observer position, supposed to be at z~0.8
robs = LB/2*np.array([1,1,0]) - cosmo.comoving_distance(z0)*np.array([0,0,1])

# calculate angular bin
dr = np.deg2rad(dthet.value) * cosmo.comoving_transverse_distance(z0)
dr_bin = np.arange(-LB.value/2, LB.value/2, dr.value)
mid_dr = 0.5*(dr_bin[1:] + dr_bin[:-1])

V_cell =  dr * dr* (cosmo.comoving_distance(z_bin[1:]) - cosmo.comoving_distance(z_bin[:-1]))
np.savetxt('%sz_bin.txt' %path_out, np.array([z_bin[:-1], z_bin[1:], mid_z, z2nu(mid_z), V_cell]).T, fmt='%.3f\t%.3f\t%.5f\t%.3f\t%.3f', header='z_l\tz_r\t\tz_mid\tnu [MHz]\tV_cell [Mpc^3]')
np.savetxt('%sdr_bin.txt' %path_out, np.array([dr_bin[:-1], dr_bin[1:], mid_dr]).T, fmt='%.5e', header='from observer coordinate system r_obs=%s\ndr_l\tdr_r\tdr_mid' %(str(robs)))

for i in tqdm(range(i_start, i_end)):
    i_z0, z0 = idx_z[i], redshift[i]
    fname = '%slc_z%d_%.1f%s_%d%s' %(path_out, i_z0, dthet.value, dthet.unit, dnu.value, dnu.unit)
    
    if not (os.path.exists(fname)):
        lc = np.zeros((mid_z.size, mid_dr.size, mid_dr.size))
        for j in tqdm(range(100)):
            # get halo variables and ID for the redshift z file (mvir, rvir, rx, ry, rz, vx, vy, vz)
            mvir, _, rx, ry, rz, _, _, _ = read_cbin('%shalodir_%03d/halolist_z%.2f_%d.bin' %(path, i_z0, z0, j), dimensions=2)
            
            for i_z, z in enumerate(mid_z):
                # get HI mass in halos (model A in Modi+ 2019)
                mHI = MHI_Modi2019(Mh=mvir, z=z, model='A', cosmo=cosmo)

                # get only halos that contain hydrogen
                mask_HI = mHI != 0

                # calculate the line of sight distance for the halos
                rz_obs = rz[mask_HI] / cosmo.h * u.Mpc - robs[2]

                # get only halo within the redshift bin value
                mask_z = (rz_obs >= cosmo.comoving_distance(z_bin[i_z])) * (rz_obs < cosmo.comoving_distance(z_bin[i_z+1]))

                # get halos position
                #rh = np.vstack((rx[mask_HI][mask_z], np.vstack((ry[mask_HI][mask_z], rz[mask_HI][mask_z])))).T / cosmo.h * u.Mpc
                rh = np.vstack((rx[mask_HI][mask_z], ry[mask_HI][mask_z])).T / cosmo.h * u.Mpc

                # get halos position from observer POV
                rh_obs = rh - robs[:2]

                # get HI mass in redshift bin
                mHI = mHI[mask_HI][mask_z]

                ret = stats.binned_statistic_2d(rh_obs[:,0], rh_obs[:,1], mHI, 'sum', bins=[dr_bin, dr_bin])

                if(j == 0 and i_z == 0):
                    lc[i_z] = ret.statistic #(ret.statistic * u.Msun / V_cell).cgs.value
                else:
                    lc[i_z] += ret.statistic
            
            np.save(fname, lc)
    else:
        print(' %s exist already - skip' %fname)
