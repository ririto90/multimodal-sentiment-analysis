#!/bin/bash -l

#SBATCH --job-name=SIMPLE-F3v3    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-12:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=16g         # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F3v3/2025-03-09/sub-1/010_Mar-09-2025_03:43_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F3v3/2025-03-09/sub-1/010_Mar-09-2025_03:43_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F3v3/2025-03-09/sub-1/010_Mar-09-2025_03:43_PM"
fusion="multiattfusion"
dataset="MOA-MVSA-single"
lr="0.001"
dr="0.5"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Dataset: MOA-MVSA-single"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/SIMPLE-F3v3/src/ \
python -u -Wd Models/SIMPLE-F3v3/src/train.py \
    --model_fusion "multiattfusion" \
    --dataset "MOA-MVSA-single" \
    --num_epoch "50" \
    --batch_size "64" \
    --log_step 60 \
    --learning_rate "0.001" \
    --dropout_rate "0.5" \
    --weight_decay 0
