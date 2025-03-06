#!/bin/bash -l

#SBATCH --job-name=MOASC    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=debug      # Run on tier3
#SBATCH --time=0-01:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=16g         # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC/2025-03-02/sub-1/005_Mar-02-2025_11:40_AM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC/2025-03-02/sub-1/005_Mar-02-2025_11:40_AM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC/2025-03-02/sub-1/005_Mar-02-2025_11:40_AM"
fusion=""
dataset=""
lr=""
dr=""

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo "SLURM Job ID: $SLURM_JOB_ID"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/MOASC \
python -u -Wd Models/MOASC/run_project.py \

