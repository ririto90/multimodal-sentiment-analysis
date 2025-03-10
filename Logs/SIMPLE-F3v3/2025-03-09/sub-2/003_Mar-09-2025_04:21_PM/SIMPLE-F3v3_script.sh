#!/bin/bash -l

#SBATCH --job-name=SIMPLE-F3v3    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-08:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=128g              # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F3v3/2025-03-09/sub-2/003_Mar-09-2025_04:21_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F3v3/2025-03-09/sub-2/003_Mar-09-2025_04:21_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F3v3/2025-03-09/sub-2/003_Mar-09-2025_04:21_PM"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo SIMPLE-F3v3_multiattfusion_MOA-MVSA-single_lr0.0005_dr0.5
echo "SLURM Job ID: $SLURM_JOB_ID"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/SIMPLE-F3v3/src/ \
python -u -Wd Models/SIMPLE-F3v3/src/train.py \
    --model_fusion "${fusion}" \
    --dataset "${dataset}" \
    --num_epoch 25 \
    --batch_size 128 \
    --log_step 60 \
    --learning_rate "${lr}" \
    --dropout_rate "${dr}" \
    --hidden_dim "${hidden_dim}" \
    --weight_decay 0 
