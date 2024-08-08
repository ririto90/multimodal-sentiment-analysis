#!/bin/bash -l

#SBATCH --job-name=ESAFN_model    # Name of your job
#SBATCH --account=multisass               # Your Slurm account
#SBATCH --partition=tier3                 # Run on tier3
#SBATCH --time=0-04:00:00                 # 4 hours time limit
#SBATCH --nodes=1                         # # of nodes
#SBATCH --ntasks=1                        # 1 task (i.e. process)
#SBATCH --mem=32g                         # Increase RAM to 16GB
#SBATCH --gres=gpu:a100:2                 # 1 a100 GPU
#SBATCH --output=Logs/ESAFN/012_Jul-24-2024_12:54_AM/%x_%j.out           # Output file
#SBATCH --error=Logs/ESAFN/012_Jul-24-2024_12:54_AM/%x_%j.err             # Error file

# Load necessary environment
spack env activate default-ml-x86_64-24071101

# Run the main script
cd ../ESAFN
stdbuf -oL -eL bash run.sh
