import numpy as np, os, h5py
from other_utils import Timer, save_cbin

from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

timer = Timer()
timer.start()

# see: http://skiesanduniverses.org/Simulations/Uchuu/UchuuDR1Products/
path_download = 'ftp://skun.iaa.es/SUsimulations/UchuuDR1/Uchuu/RockstarExtended/'
path_out = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'

try:
    os.mkdir(path_out)
except:
    pass

# read redshift of Ushuu simulation
#idx, redshift = np.loadtxt(path_out+'redshift.txt', unpack=True, usecols=(0, 1))
#idx_dir, redshift = np.loadtxt(path_out+'ushuu_redshift.txt', unpack=True, usecols=(0, 1))
idx_dir, redshift = [37], [0.70]

# Processor repartition based on the rank
#numbers = np.arange(100)
snaps = np.arange(100)
numbers = snaps[np.where(snaps % nprocs == rank)[0]]
print(' rank=%d ---> %s' %(rank, str(numbers)))

for i, z in zip(idx_dir, redshift):
    str_z = ('%.2f' %z).replace('.', 'p')
 
    #if (not os.path.exists('%shalodir_%03d' %(path_out, i)) and rank == 0):
    try:
        os.mkdir('%shalodir_%03d' %(path_out, i))
    except:
        pass

    os.chdir('%shalodir_%03d' %(path_out, i))
    
    for j in numbers:
        fname_h5 = '%shalodir_%03d/halolist_z%s_%d.h5' %(path_out, i, str_z, j)
        fname_id = '%shalodir_%03d/haloid_z%.2f_%d.bin' %(path_out, i, z, j)
        fname_halo = '%shalodir_%03d/halolist_z%.2f_%d.bin' %(path_out, i, z, j)

        if not (os.path.exists(fname_id) and os.path.exists(fname_halo)):
            if not (os.path.exists(fname_h5)):
                # download
                os.system('wget %shalodir_%03d/halolist_z%s_%d.h5' %(path_download, i, str_z, j))
                
            # open h5 data
            hf = h5py.File(fname_h5, 'r')
            h_pid = np.array(hf['pid'])

            # get only parent (distinct) halos
            mask_halo = h_pid == -1

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
            
            # save trim data
            save_cbin(filename=fname_halo, data=[mvir, rvir, r_x, r_y, r_z, v_x, v_y, v_z], datatype=np.float32)
            save_cbin(filename=fname_id, data=idx, datatype=np.int64)

            # delete full data
            os.system('rm %s' %fname_h5)
            timer.lap('done %d, %d' %(i, j))
        else:
            print(' File halodir_%03d/halolist_z%.2f_%d.bin exist: SKIP' %(i, z, j))
        
timer.stop()

