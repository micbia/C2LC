import numpy as np, h5py, pickle
from other_utils import Timer, save_cbin


t = Timer()
path = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'
fname_in = 'halodir_035/halolist_z0p86_27.h5'

#path = '/work/ska/HIRAX/UchuuSim/MiniUchuu/RockstarExtended/'
#fname_in = 'MiniUchuu_halolist_z1p03.h5'
#fname_in = 'MiniUchuu_halolist_z2p46.h5'


t.start()
hf = h5py.File(path+fname_in, 'r')
#print(hf.keys())

# get halos IDs
h_pid = np.array(hf['pid'])

# get only parent (distinct) halos
mask_halo = h_pid == -1

# see for units: http://skiesanduniverses.org/Simulations/Uchuu/HaloCatalogues/
idx = np.array(hf['id'])[mask_halo]  # in Msun/h
mvir = np.array(hf['Mvir'])[mask_halo]  # in Msun/h
rvir = np.array(hf['Rvir'])[mask_halo]  # in comoving kpc/h
r_x = np.array(hf['x'])[mask_halo]     # in comoving Mpc/h
r_y = np.array(hf['y'])[mask_halo]
r_z = np.array(hf['z'])[mask_halo]
v_x = np.array(hf['vx'])[mask_halo]    # in physical km/s (peculiar)
v_y = np.array(hf['vy'])[mask_halo]
v_z = np.array(hf['vz'])[mask_halo]
hf.close()
t.lap('closed h5 file')

print(' %.2f %% of halo in catalogues are parent halos.' %(100*(np.count_nonzero(mask_halo)/mask_halo.size)))

# store data
fname = fname_in[fname_in.rfind('halo'):-3].replace('p', '.') + '.bin'
save_cbin(filename=path+fname, data=[mvir, rvir, r_x, r_y, r_z, v_x, v_y, v_z], datatype=np.float32)
save_cbin(filename=path+'haloid_z0.86_27.bin', data=idx, datatype=np.int64)
print([mvir, rvir, r_x, r_y, r_z, v_x, v_y, v_z])
print(idx)
t.stop()