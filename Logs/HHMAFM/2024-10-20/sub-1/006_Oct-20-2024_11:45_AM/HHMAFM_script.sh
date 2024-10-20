#!/bin/bash -l

#SBATCH --job-name=HHMAFM    # Name of your job
#SBATCH --account=multisass         # Your Slurm account
#SBATCH --partition=debug           # Run on tier3
#SBATCH --time=0-12:00:00           # 12 hours time limit
#SBATCH --nodes=1                   # # of nodes
#SBATCH --ntasks=1                  # 1 task (i.e. process)
#SBATCH --mem=32g                   # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:2           # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-20/006_Oct-20-2024_11:45_AM/%x_%j.out     # Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-20/006_Oct-20-2024_11:45_AM/%x_%j.err       # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variable (explicitly within the script) for access within the python script environment
export NEW_LOGS_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-20/006_Oct-20-2024_11:45_AM"
export REPO_DIR="/home/rgg2706/Multimodal-Sentiment-Analysis"

# Run the main script
cd /home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-20/006_Oct-20-2024_11:45_AM
echo "> NEW_LOG_DIR: /home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-20/006_Oct-20-2024_11:45_AM"
echo "Current directory: $(pwd)"

stdbuf -oL -eL bash run.sh
