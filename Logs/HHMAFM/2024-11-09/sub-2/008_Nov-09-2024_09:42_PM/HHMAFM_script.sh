#!/bin/bash -l

#SBATCH --job-name=HHMAFM    # Name of your job
#SBATCH --account=multisass         # Your Slurm account
#SBATCH --partition=tier3           # Run on tier3
#SBATCH --time=0-12:00:00           # 12 hours time limit
#SBATCH --nodes=1                   # # of nodes
#SBATCH --ntasks=1                  # 1 task (i.e. process)
#SBATCH --mem=32g                   # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:1           # 2 a100 GPUs
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-09/sub-2/008_Nov-09-2024_09:42_PM/%x_%j.out     # Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-09/sub-2/008_Nov-09-2024_09:42_PM/%x_%j.err       # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variable (explicitly within the script) for access within the python script environment
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-09/sub-2/008_Nov-09-2024_09:42_PM"
export REPO_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis"

# Run the main script
cd "/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-11-09/sub-2/008_Nov-09-2024_09:42_PM"

stdbuf -oL -eL bash run.sh
