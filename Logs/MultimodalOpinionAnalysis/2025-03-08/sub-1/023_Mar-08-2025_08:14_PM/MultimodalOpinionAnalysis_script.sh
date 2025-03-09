#!/bin/bash -l

#SBATCH --job-name=MultimodalOpinionAnalysis    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=debug      # Run on tier3
#SBATCH --time=0-01:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=32g         # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MultimodalOpinionAnalysis/2025-03-08/sub-1/023_Mar-08-2025_08:14_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MultimodalOpinionAnalysis/2025-03-08/sub-1/023_Mar-08-2025_08:14_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MultimodalOpinionAnalysis/2025-03-08/sub-1/023_Mar-08-2025_08:14_PM"
fusion=""
dataset=""
lr=""
dr=""

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo "SLURM Job ID: $SLURM_JOB_ID"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/MultimodalOpinionAnalysis \
python -u -Wd Models/MultimodalOpinionAnalysis/run_project.py \

