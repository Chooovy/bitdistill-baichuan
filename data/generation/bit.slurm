#! /bin/bash
#SBATCH --job-name=guhao-bitdistill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --partition=gov
#SBATCH --gres=gpu:1 # Adjust number of GPUs here

#SBATCH --time=2-00:00:00
#SBATCH --output=./slurm_log/%x-%j-512.out
#SBATCH --err=./slurm_log/%x-%j-512.err
#SBATCH --chdir=/home/lilujun/workspace/BitDistiller/data/generation
source /home/lilujun/workspace_tianyu/miniconda3/bin/activate  baichuan
# conda activate baichuan
bash /home/lilujun/workspace/BitDistiller/data/generation/generate.sh
# srun 