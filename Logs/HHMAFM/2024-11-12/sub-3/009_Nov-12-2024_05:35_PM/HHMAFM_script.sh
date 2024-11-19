#!/bin/bash -l

#SBATCH --job-name=HHMAFM    # Name of your job
#SBATCH --account=multisass    # Your Slurm account
#SBATCH --partition=tier3      # Run on tier3
#SBATCH --time=0-12:00:00       # 4 hours time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # 1 task (i.e., process)
#SBATCH --mem=96g              # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:2      # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-12/sub-3/009_Nov-12-2024_05:35_PM/%x_%j.out# Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-12/sub-3/009_Nov-12-2024_05:35_PM/%x_%j.err  # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-12/sub-3/009_Nov-12-2024_05:35_PM"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis"

echo HHMAFM_mfcchfusion3_mvsa-mts-oversampled_lr0.0003_dr0.2
echo "SLURM Job ID: $SLURM_JOB_ID"

PYTHONPATH=$PYTHONPATH:${REPO_DIR}/Models/HHMAFM/src/ \
python -u -Wd Models/HHMAFM/src/train.py \
    --model_fusion "${fusion}" \
    --dataset "${dataset}" \
    --num_epoch 40 \
    --batch_size 128 \
    --log_step 60 \
    --learning_rate "${lr}" \
    --dropout_rate "${dr}" \
    --hidden_dim 768 \
    --weight_decay 0 
