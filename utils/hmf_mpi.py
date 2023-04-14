import numpy as np, matplotlib.pyplot as plt
from other_utils import read_cbin

from tqdm import tqdm
from scipy import stats
from colossus.cosmology import cosmology
from colossus.lss import mass_function

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

path = '/work/ska/HIRAX/UchuuSim/Uchuu/RockstarExtended/'
path_out = '/home/mbianco/codes/hirax/output_sim/'

cosmo = cosmology.setCosmology('planck15')
rho0 = cosmo.rho_c(z=0)*cosmo.Om0

# Set loop starting index per processor and task
idx_z, redshift = np.loadtxt('%sredshift.txt' %path, usecols=(0,1), unpack=True)
jobs_proc = np.arange(0, redshift.size)
range_proc = jobs_proc[jobs_proc % nprocs == rank]
print(' rank=%d has tasks\n halodir : %s' %(rank, str(range_proc)))

massbin = np.logspace(9, 15, 21)
M, dM = (massbin[:-1] + massbin[1:])*0.5, massbin[1:]-massbin[:-1]

# Ushuu data
Lbox = 2000. # cMpc/h
#i_z0, z0 = 33, 1.03
#i_z0, z0 = 24, 2.03
#i_z0, z0 = 30, 1.32

min_mass = np.inf
max_mass = 0.

# calculate Ushuu HMF
for i in range_proc:
    i_z0, z0 = idx_z[i], redshift[i]
    for j in tqdm(range(0, 100)):
        mvir, _, rx, ry, rz, _, _, _ = read_cbin('%shalodir_%03d/halolist_z%.2f_%d.bin' %(path, i_z0, z0, j), dimensions=2)
        if(j == 0):
            Nhalo = stats.binned_statistic(x=mvir, values=None, statistic='count', bins=massbin).statistic
        else:
            Nhalo += stats.binned_statistic(x=mvir, values=None, statistic='count', bins=massbin).statistic

        min_mass = np.min([min_mass, mvir.min()])
        max_mass = np.max([max_mass, mvir.max()])

    #print('%e\t%e' %(min_mass, max_mass))

    dn = Nhalo / (Lbox**3)
    #hmf_sim = dn/dM * M**2 / rho0/1e9
    hmf_sim = dn/(dM/M) 

    # analytical HMF
    hmf_anal = mass_function.massFunction(M, z0, q_out='dndlnM', mdef='fof', model='crocce10') #tinker08, press74

    # PLOTS
    plt.loglog(M, hmf_sim, label='Ushuu Simulation')
    plt.loglog(M, hmf_anal, label='Crocce et al. (2010)')
    plt.title('z = %.3f' %z0)
    plt.xlabel(r'$M$  $[M_{\odot}/h]$'), plt.ylabel(r'$dN/d\,ln(M)$  $[h^3\,Mpc^{-3}]$')
    plt.legend()
    plt.savefig('%sushuu_hmf_z%.3f.png' %(path_out, z0), bbox_inches='tight', facecolor='white')
    plt.clf()

    np.savetxt('%shmf_ushuu_z%.3f.txt' %(path_out, z0), np.array([M, hmf_sim, hmf_anal]).T, header='%e\t%e\nM\thmf_sim\thmf_anal' %(min_mass, max_mass))