import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from tqdm import tqdm

import astropy.units as u
import astropy.constants as cst
from astropy.cosmology import Planck15 as cosmo

from utils.utils_cosm import ExtendCosmology

print("___  ___      _          _     _       _     _                       \n|  \/  |     | |        | |   (_)     | |   | |                      \n| .  . | __ _| | _____  | |    _  __ _| |__ | |_ ___ ___  _ __   ___ \n| |\/| |/ _` | |/ / _ \ | |   | |/ _` | '_ \| __/ __/ _ \| '_ \ / _ \ \n| |  | | (_| |   <  __/ | |___| | (_| | | | | || (_| (_) | | | |  __/\n\_|  |_/\__,_|_|\_\___| \_____/_|\__, |_| |_|\__\___\___/|_| |_|\___|\n                                  __/ |                              \n                                 |___/                               ")

cosmo = ExtendCosmology(cosmo)

# ----------------------------------------------------------------------------------------
path_in = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/MHI_Padmanabhan/'
path_out = path_in
dthet, dnu = 0.4 * u.deg, 390.*u.kHz      # for HIRAX
interpolation = 'sigmoid'
# ----------------------------------------------------------------------------------------

typ_model = path_in[path_in[:-1].rfind('/')+1:-1]
fout_lc = '%slc_dT_%s_%.1f%s_%d%s.npy' %(path_out, typ_model, dthet.value, dthet.unit, dnu.value, dnu.unit)

idx_z, cube_redshift = np.loadtxt('%sredshift.txt' %(path_in), usecols=(0,1), unpack=True)
lc_redshift, Vcell = np.loadtxt('%sz_bin.txt' %(path_in), usecols=(2, 4), unpack=True)
idx_z, cube_redshift = idx_z[::-1], cube_redshift[::-1]
Vcell *= u.Mpc**3
angl = cosmo.cdist2deg(np.loadtxt('%sdr_bin.txt' %(path_in), usecols=(2))*u.Mpc, lc_redshift.min())
FoV = (angl.max()-angl.min())
angl = angl.value
print('\nz = [%.3f, %.3f]\t FoV = %.2f %s' % (lc_redshift.min(), lc_redshift.max(), FoV.value, FoV.unit))

# calculate dTb from halo mass files
for i_z0, z0 in zip(idx_z, cube_redshift):
    if not (os.path.exists('%sdT_z%.2f_%.1f%s_%d%s.npy' %(path_out, z0, dthet.value, dthet.unit, dnu.value, dnu.unit))):
        # get HI mass
        mHI = np.load('%slc_z%d_%.1f%s_%d%s.npy' %(path_out, i_z0, dthet.value, dthet.unit, dnu.value, dnu.unit)) * u.Msun

        # calculate the fraction of neutral HI
        xHI = (mHI / Vcell / (cosmo.mean_molecular * cst.m_p * cosmo.nH0)).cgs.value

        # calculate differential brightness
        dT = cosmo.dTb(xHI, z=z0)

        np.save('%sdT_z%.2f_%.1f%s_%d%s.npy' %(path_out, z0, dthet.value, dthet.unit, dnu.value, dnu.unit), dT.value) # in mK

######## Make lightcone ########
lightcone = np.zeros((angl.size, angl.size, lc_redshift.size), dtype='float32')

# Make the lightcone, one slice at a time
if not (os.path.exists(fout_lc)):
    for i_l in tqdm(range(lc_redshift.size)):
        z = lc_redshift[i_l]

        i_cube = np.digitize(z, cube_redshift, right=True)
        z_low, z_high = cube_redshift[i_cube-1], cube_redshift[i_cube]

        data_low = np.load('%sdT_z%.2f_%.1f%s_%d%s.npy' %(path_out, z_low, dthet.value, dthet.unit, dnu.value, dnu.unit), mmap_mode='r') # in mK
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
else:
    lightcone = np.load(fout_lc)

# save lightcone
np.save(fout_lc, lightcone)

i_z, i_lc = lightcone.shape[2]//2, lightcone.shape[1]//2

fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])

ax0 = fig.add_subplot(gs[0,0])
im = ax0.pcolormesh(lc_redshift, angl, lightcone[i_lc], cmap='jet')#, norm=LogNorm())
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

