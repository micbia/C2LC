import numpy as np
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

buffstart = comm.gather(True, root=0)
if(rank == 0):
    print(buffstart, all(buffstart))
    print("___  ___      _          _     _       _     _                       \n|  \/  |     | |        | |   (_)     | |   | |                      \n| .  . | __ _| | _____  | |    _  __ _| |__ | |_ ___ ___  _ __   ___ \n| |\/| |/ _` | |/ / _ \ | |   | |/ _` | '_ \| __/ __/ _ \| '_ \ / _ \ \n| |  | | (_| |   <  __/ | |___| | (_| | | | | || (_| (_) | | | |  __/\n\_|  |_/\__,_|_|\_\___| \_____/_|\__, |_| |_|\__\___\___/|_| |_|\___|\n                                  __/ |                              \n                                 |___/                               ")

jobs_proc = np.arange(0, 12)
range_proc = jobs_proc[jobs_proc % nprocs == rank]

if(rank == 0):
    count = 0
    lc = np.zeros((jobs_proc.size, 2))

for i in range_proc:
    data = np.zeros((2))+rank

    new_data = comm.gather(data, root=0)
    idx = comm.gather(i, root=0)

    if(rank == 0):
        print(' Gather = %d\t' %count, new_data)
        print(np.shape(new_data))
        print(idx, '\n')
        count += 1
        for i_n, i_z in enumerate(idx):
            lc[i_z] = new_data[i_n]

comm.Barrier()

if(rank == 0):
    print('Final:')
    print(lc)

