import os, numpy as np

path_download = 'https://users.flatironinstitute.org/~fvillaescusa/M_HI/'
path_out = '/work/ska/HIRAX/CAMELS_dataset/MHI_grids/'

box_size, npart = [75, 205], [1820, 2500]
redshift = [0.997, 1.496, 2.002]

os.chdir(path_out)
for z in redshift:
    for LB, NP in zip(box_size, npart):
        if not (os.path.exists('%sM_HI_%d_%d_z=%.3f.hdf5' %(path_out, LB, NP, z))):
            os.system('wget %sM_HI_%d_%d_z=%.3f.hdf5' %(path_download, LB, NP, z))

