#!/bin/bash -l
#SBATCH --job-name=make_lc
#SBATCH --account=lastro-astrosignals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.bianco@epfl.ch
##SBATCH -t 01:30:00
##SBATCH -t 00:40:00
#SBATCH -t 00:15:00
#SBATCH --output=logs/lc.%J.out
#SBATCH --error=logs/lc.%J.err

module purge 
module load intel/19.0.5 intel-mpi/2019.5.281
module load py-mpi4py/3.0.3
module load python/3.7.7

srun -n $SLURM_NTASKS python make_lightcone.py
#srun -n $SLURM_NTASKS python utils/hmf_mpi.py
#srun -n $SLURM_NTASKS python utils/trim_ushuu_halo.py