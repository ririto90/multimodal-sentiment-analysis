#!/bin/bash -l

#SBATCH --job-name=DMLAN    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-08:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=64g              # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/DMLAN/2024-11-12/sub-1/001_Nov-12-2024_12:04_AM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/DMLAN/2024-11-12/sub-1/001_Nov-12-2024_12:04_AM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/DMLAN/2024-11-12/sub-1/001_Nov-12-2024_12:04_AM"
fusion="dmlanfusion"
dataset="mvsa-mts"
lr="0.0001"
dr="0.5"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo "SLURM Job ID: $SLURM_JOB_ID"

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/DMLAN/src/ \
python -u -Wd Models/DMLAN/src/train.py \
    --model_fusion "${fusion}" \
    --dataset "${dataset}" \
    --num_epoch 50 \
    --batch_size 64 \
    --log_step 60 \
    --learning_rate "${lr}" \
    --dropout_rate "${dr}" \
    --weight_decay 0
