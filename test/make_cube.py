import numpy as np
import astropy.units as u
import astropy.constants as cst

from tqdm import tqdm
from scipy import stats
from astropy.cosmology import Planck15 as cosmo
from utils.read_halo import read_cbin
from utils.utils_cosm import cdist2z, z2nu, cdist2deg, MHI

path_in = '/work/backup/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'
path_out = path_in + 'outputs/'
dthet, dnu = 0.5 * u.deg, 5.*u.MHz

# h = 0.6774
# Mvir in Msun/h
# Rvir in comoving kpc/h
# rx, ry and rz in comoving Mpc/h
# vx, vy, and vz in physical km/s (peculiar velocity)

# define boxsize and first redshift
LB = 2000./cosmo.h * u.Mpc
idx_z, redshift = np.loadtxt(path_in+'redshift.txt', usecols=(0,1), unpack=True)
i_z0, z0 = 35, 0.86

# get observer position, supposed to be at z=0
robs = LB/2*np.array([1,1,0]) - cosmo.comoving_distance(z0)*np.array([0,0,1])

# calculate redshift bin
nu0 = 1.42*u.GHz
zi, zi1 = redshift.min(), 0
z_bin = [zi]

while zi1 <= redshift.max():
    zi1 = ((zi + dnu/nu0 * (1+zi)) / (1-dnu/nu0*(1+zi))).value
    z_bin.append(zi1)
    #print(zi, zi1, np.round(z2nu(zi) - z2nu(zi1)))
    zi = zi1
z_bin = np.array(z_bin)
mid_z = 0.5*(z_bin[1:] + z_bin[:-1])

# calculate angular bin
dr = np.deg2rad(dthet.value) * cosmo.comoving_transverse_distance(redshift.min())
dr_bin = np.arange(-LB.value/2, LB.value/2, dr.value)
mid_dr = 0.5*(dr_bin[1:] + dr_bin[:-1])

V_cell =  dr * dr* (cosmo.comoving_distance(z_bin[1:]) - cosmo.comoving_distance(z_bin[:-1]))
np.savetxt('%sz%d_bin.txt' %(path_out, i_z0), np.array([z_bin[:-1], z_bin[1:], mid_z, z2nu(mid_z), V_cell]).T, fmt='%.3f\t%.3f\t%.5f\t%.3f\t%.3f', header='z_l\tz_r\t\tz_mid\tnu [MHz]\tV_cell [Mpc^3]')
np.savetxt('%sdr%d_bin.txt' %(path_out, i_z0), np.array([dr_bin[:-1], dr_bin[1:], mid_dr]).T, fmt='%.5e', header='from observer coordinate system r_obs=%s\ndr_l\tdr_r\tdr_mid' %(str(robs)))

lc = np.zeros((mid_z.size, mid_dr.size, mid_dr.size))
for j in tqdm(range(100)):
    # get halo variables and ID for the redshift z0 file (mvir, rvir, rx, ry, rz, vx, vy, vz)
    mvir, _, rx, ry, rz, _, _, _ = read_cbin('%shalodir_%03d/halolist_z%.2f_%d.bin' %(path_in, i_z0, z0, j), dimensions=2)
    idx = read_cbin('%shalodir_%03d/haloid_z%.2f_%d.bin' %(path_in, i_z0, z0, j), dimensions=1, datatype=np.int64)
    
    for i_z, z in enumerate(mid_z):
        # get HI mass in halos (model A in Modi+ 2019)
        mHI = MHI(Mh=mvir, z=z, cosmo=cosmo)

        # get only halos that contain hydrogen
        mask_HI = mHI != 0

        # calculate the line of sight distance for the halos
        rz_obs = rz[mask_HI] / cosmo.h * u.Mpc - robs[2]

        # get the redshift corresponding to the LOS distance
        #z_corresp_to_rz = cdist2z(dist=rz_obs.value, cosmo=cosmo)

        # get only halo within the redshift bin value
        #mask_z = (z_corresp_to_rz >= z_bin[i_z]) * (z_corresp_to_rz < z_bin[i_z+1])
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
    
    np.save('%slc_z%d_%.1f%s_%d%s' %(path_out, i_z0, dthet.value, dthet.unit, dnu.value, dnu.unit), lc)
