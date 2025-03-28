#!/bin/bash -l

#SBATCH --job-name=SIMPLE-F2    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=debug      # Run on tier3
#SBATCH --time=0-12:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=16g         # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F2/2025-02-13/sub-1/002_Feb-13-2025_12:56_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F2/2025-02-13/sub-1/002_Feb-13-2025_12:56_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F2/2025-02-13/sub-1/002_Feb-13-2025_12:56_PM"
fusion="simplefusion"
dataset="mvsa-mts-v3-30"
lr="0.001"
dr="0.5"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo "SLURM Job ID: $SLURM_JOB_ID"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/SIMPLE-F2/src/ \
python -u -Wd Models/SIMPLE-F2/src/train.py \
    --model_fusion "${fusion}" \
    --dataset "${dataset}" \
    --num_epoch 200 \
    --batch_size 256 \
    --log_step 16 \
    --learning_rate "${lr}" \
    --dropout_rate "${dr}" \
    --weight_decay 0
