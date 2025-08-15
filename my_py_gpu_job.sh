#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j-%N_slurm.out
#SBATCH --error=R-%x.%j.err
## Activate right env

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load FFTW.MPI/3.3.10-gompi-2022a
module load GSL/2.7-GCC-11.3.0
module load HDF5/1.12.2-gompi-2022a
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
module load UCX/1.12.1-GCCcore-11.3.0
export OMP_NUM_THREADS=18

cd /home/osavchenko/EB-DISCO
srun python3 os250814-discoeb_training_data_generation_custom_k.py #train_cpj_camb_cmb.py #_training_data.py #os250804-discoeb_training_data_generation.py
