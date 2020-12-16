#!/bin/bash
#SBATCH --job-name=hpc_task        # Job name
#SBATCH --error=task.err           # File for outputting error
#SBATCH --output=task.log          # File for outputting results
#SBATCH --time=72:00:00            # Maximum execution time
#SBATCH --ntasks=1                 # Number of MPI processes
#SBATCH --nodes=1                  # Required number of nodes
#SBATCH --gpus=1                   # Required GPU
#SBATCH --cpus-per-task=1          # Required CPU number

module load Python/Anaconda_v10.2019 CUDA/10.2
srun bash job_hpc_evaluator.sh
