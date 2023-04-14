#import tools21cm as t2c
import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL

path_in = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'
path_out = '/home/mbianco/codes/hirax/output_sim/'
#lc_HI = np.load('lc_dT_MHI_Modi2019_ModelC_0.4deg_390kHz.npy')

BoxSize = 2000.0  #Mpc/h
wanted_z = 0.9
#fname = 'Padmanabhan'
fname = 'Modi2019_ModelC'

lc_redshift, Vcell = np.loadtxt('%sMHI_%s/z_bin.txt' %(path_in, fname), usecols=(2, 4), unpack=True)
i_slice = np.argmin(np.abs(lc_redshift - wanted_z))
z = lc_redshift[i_slice]

dT_model = np.load('%sMHI_%s/lc_dT_MHI_%s_0.4deg_390kHz.npy' %(path_in, fname, fname),)[...,i_slice]
dT_mcut = np.load('%sMHI_%s_M10cut/lc_dT_MHI_%s_M10cut_0.4deg_390kHz.npy' %(path_in, fname, fname))[...,i_slice]
dT_rsd = np.load('%sMHI_%s_rsd/lc_dT_MHI_%s_rsd_0.4deg_390kHz.npy' %(path_in, fname, fname))[...,i_slice]

# Pk in (Mpc/h)^2 and k in h/Mpc
Pk2D_model = PKL.Pk_plane(dT_model.astype(np.float32), BoxSize, 'None', 1)
Pk2D_mcut = PKL.Pk_plane(dT_mcut.astype(np.float32), BoxSize, 'None', 1)
Pk2D_rsd = PKL.Pk_plane(dT_rsd.astype(np.float32), BoxSize, 'None', 1)

plt.plot(Pk2D_model.k, Pk2D_model.Pk, color='tab:blue', label=fname)
plt.plot(Pk2D_mcut.k, Pk2D_mcut.Pk, color='tab:orange', label='M10cut')
plt.plot(Pk2D_rsd.k, Pk2D_rsd.Pk, color='tab:red', label='rsd')
plt.xscale('log'), plt.yscale('log')

plt.legend()
plt.title('z=%.3f' %z)
plt.xlabel(r'k $[h/Mpc]$'), plt.ylabel(r'$P_k$ [$(Mpc/h)^2$]')
plt.savefig('%spk_%s_z%.3f.png' %(path_out, fname, z), bbox_inches='tight'), plt.clf()
