import numpy as np

def read_cbin(filename, datatype=np.float32, order='C', dimensions=3):
    f = open(filename)
    header = np.fromfile(f, count=dimensions, dtype='int32')
    data = np.fromfile(f, dtype=datatype, count=np.prod(header))
    data = data.reshape(header, order=order)
    f.close()
    return data

#path = '/work/backup/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/halodir_035/'

# h = 0.6774
# Mvir in Msun/h
# Rvir in comoving kpc/h
# rx, ry and rz in comoving Mpc/h
# vx, vy, and vz in physical km/s (peculiar velocity)

#mvir, rvir, rx, ry, rz, vx, vy, vz = read_cbin(path+'halolist_z0.86_27.bin', dimensions=2)
#idx = read_cbin(path+'haloid_z0.86_27.bin', dimensions=1, datatype=np.int64)
