#!/bin/bash -l

#SBATCH --job-name=SIMPLE-F2    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-08:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=128g              # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F2/2025-02-13/sub-2/004_Feb-13-2025_01:10_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F2/2025-02-13/sub-2/004_Feb-13-2025_01:10_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/SIMPLE-F2/2025-02-13/sub-2/004_Feb-13-2025_01:10_PM"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo SIMPLE-F2_simplefusion_mvsa-mts-v3_lr0.0001_dr0.3
echo "SLURM Job ID: $SLURM_JOB_ID"

export PYTHONPATH=:/home/rgg2706/Multimodal-Sentiment-Analysis

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/SIMPLE-F2/src/ \
python -u -Wd Models/SIMPLE-F2/src/train.py \
    --model_fusion "${fusion}" \
    --dataset "${dataset}" \
    --num_epoch 70 \
    --batch_size 128 \
    --log_step 60 \
    --learning_rate "${lr}" \
    --dropout_rate "${dr}" \
    --hidden_dim "${hidden_dim}" \
    --weight_decay 0 
