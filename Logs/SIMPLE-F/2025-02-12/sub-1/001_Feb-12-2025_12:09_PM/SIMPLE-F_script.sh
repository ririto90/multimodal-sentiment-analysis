#!/bin/bash -l

#SBATCH --job-name=SIMPLE-F    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-12:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=128g         # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F/2025-02-12/sub-1/001_Feb-12-2025_12:09_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F/2025-02-12/sub-1/001_Feb-12-2025_12:09_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F/2025-02-12/sub-1/001_Feb-12-2025_12:09_PM"
fusion="simplefusion"
dataset="mvsa-mts-v3"
lr="0.0001"
dr="0.5"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo "SLURM Job ID: $SLURM_JOB_ID"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/SIMPLE-F/src/ \
python -u -Wd Models/SIMPLE-F/src/train.py \
    --model_fusion "simplefusion" \
    --dataset "mvsa-mts-v3" \
    --num_epoch "100" \
    --batch_size "64" \
    --log_step 60 \
    --learning_rate "0.0001" \
    --dropout_rate "0.5" \
    --weight_decay 0
