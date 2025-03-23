#!/bin/bash -l

#SBATCH --job-name=MOASC4    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-01:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=16g         # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC4/2025-03-19/sub-1/003_Mar-19-2025_11:54_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC4/2025-03-19/sub-1/003_Mar-19-2025_11:54_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC4/2025-03-19/sub-1/003_Mar-19-2025_11:54_PM"
fusion=""
dataset="MOA-MVSA-multiple"
lr=""
dr=""

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"


echo "SLURM Job ID: $SLURM_JOB_ID"
echo 'This model run an averaged macro-F1 and weighted macro-F1. This model also takes from the datasets in the Datasets folder'
echo "Model: MOASC4"
echo "Dataset: MOA-MVSA-multiple"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/MOASC4 \
python -u -Wd Models/MOASC4/run_project.py \
    --model_name "MOASC4" \
    --dataset "MOA-MVSA-multiple" \


