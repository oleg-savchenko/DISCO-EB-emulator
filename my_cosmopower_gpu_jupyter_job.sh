#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j-%N_slurm.out
## Activate right env
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0 
export TF_XLA_FLAGS=--xla_gpu_cuda_data_dir=/sw/arch/RHEL8/EB_production/2022/software/CUDA/11.8.0/nvvm/libdevice
module load IPython/8.5.0-GCCcore-11.3.0
module load jupyter-resource-usage/0.6.3-GCCcore-11.3.0
module load jupyter-server/1.21.0-GCCcore-11.3.0
module load jupyter-server-proxy/3.2.2-GCCcore-11.3.0
# module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
# module load UCX/1.12.1-GCCcore-11.3.0
source /home/osavchenko/venvs/cp/bin/activate
export OMP_NUM_THREADS=18


PORT=`shuf -i 5000-5999 -n 1`
# PORT=8897
LOGIN_HOST=${SLURM_SUBMIT_HOST}-pub.snellius.surf.nl
BATCH_HOST=$(hostname)
# ssh -N -f -R ${PORT}:localhost:${PORT} ${BATCH_HOST}
echo "To connect to the notebook type the following command from your local terminal:"
echo "ssh -J ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT}:localhost:${PORT}"
echo
echo "After connection is established in your local browser go to the address:"
echo "http://localhost:${PORT}"
jupyter notebook --no-browser --port $PORT --NotebookApp.ip='0.0.0.0' --NotebookApp.allow_origin='*' --NotebookApp.allow_remote_access=True --NotebookApp.token='' --NotebookApp.password=''

# jupyter notebook --no-browser --port $PORT --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'
