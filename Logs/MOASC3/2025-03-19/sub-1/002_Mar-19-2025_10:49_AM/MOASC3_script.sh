#!/bin/bash -l

#SBATCH --job-name=MOASC3    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-04:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=92g         # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC3/2025-03-19/sub-1/002_Mar-19-2025_10:49_AM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC3/2025-03-19/sub-1/002_Mar-19-2025_10:49_AM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/MOASC3/2025-03-19/sub-1/002_Mar-19-2025_10:49_AM"
fusion=""
dataset="MVSA-multiple"
lr=""
dr=""

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"


echo "SLURM Job ID: $SLURM_JOB_ID"
echo 'This model run adds averaged macro-F1 and weighted macro-F1'
echo "Model: MOASC3"
echo "Dataset: MVSA-multiple"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/MOASC3 \
python -u -Wd Models/MOASC3/run_project.py \
    --model_name "MOASC3" \
    --dataset "MVSA-multiple" \


