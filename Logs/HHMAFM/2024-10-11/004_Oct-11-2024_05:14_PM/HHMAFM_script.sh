#!/bin/bash -l

#SBATCH --job-name=HHMAFM    # Name of your job
#SBATCH --account=multisass         # Your Slurm account
#SBATCH --partition=debug           # Run on tier3
#SBATCH --time=0-08:00:00           # 4 hours time limit
#SBATCH --nodes=1                   # # of nodes
#SBATCH --ntasks=1                  # 1 task (i.e. process)
#SBATCH --mem=32g                   # Increase RAM to 16GB
#SBATCH --gres=gpu:a100:1           # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-11/004_Oct-11-2024_05:14_PM/%x_%j.out     # Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-11/004_Oct-11-2024_05:14_PM/%x_%j.err       # Error file

# Load necessary environment
spack env activate default-ml-x86_64-24071101

# Run the main script
cd /home/rgg2706/Multimodal-Sentiment-Analysis/Models/HHMAFM/
stdbuf -oL -eL bash /home/rgg2706/Multimodal-Sentiment-Analysis/Models/HHMAFM/run.sh
