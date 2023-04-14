#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --account=lastro-astrosignals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=michele.bianco@epfl.ch
#SBATCH -t 00:02:00

#### ONE NODE ####
#SBATCH --output=../logs/test.%J.out
#SBATCH --error=../logs/test.%J.err

#### ARRAY JOB ####
##SBATCH --array=0-1
##SBATCH --output=logs/lc.%J-%A.out
##SBATCH --error=logs/lc.%J-%A.err

module purge 
module load intel/19.0.5 intel-mpi/2019.5.281
module load py-mpi4py/3.0.3
module load python/3.7.7

srun -n $SLURM_NTASKS python test_mpi.py
#srun -n $SLURM_NTASKS python make_lightcone.py
#srun -n $SLURM_NTASKS python utils/trim_ushuu_halo.py