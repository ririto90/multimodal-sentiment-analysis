#!/bin/bash -l

#SBATCH --job-name=HHMAFM    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-4:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=32g              # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-11/sub-3/004_Nov-11-2024_06:46_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-11/sub-3/004_Nov-11-2024_06:46_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-11/sub-3/004_Nov-11-2024_06:46_PM"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo HHMAFM_mfcchfusion4_mvsa-mts-undersampled_lr0.0005_dr0.5
echo "SLURM Job ID: $SLURM_JOB_ID"

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/HHMAFM/src/ \
python -u -Wd Models/HHMAFM/src/train.py \
    --model_fusion "${fusion}" \
    --dataset "${dataset}" \
    --num_epoch 15 \
    --batch_size 64 \
    --log_step 60 \
    --learning_rate "${lr}" \
    --dropout_rate "${dr}" \
    --weight_decay 0
