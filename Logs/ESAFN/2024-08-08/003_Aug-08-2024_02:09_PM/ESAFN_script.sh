#!/bin/bash -l

#SBATCH --job-name=ESAFN    # Name of your job
#SBATCH --account=multisass         # Your Slurm account
#SBATCH --partition=tier3           # Run on tier3
#SBATCH --time=0-04:00:00           # 4 hours time limit
#SBATCH --nodes=1                   # # of nodes
#SBATCH --ntasks=1                  # 1 task (i.e. process)
#SBATCH --mem=16g                   # Increase RAM to 16GB
#SBATCH --gres=gpu:a100:1           # 1 a100 GPU
#SBATCH --output=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/ESAFN/015_Aug-08-2024_02:09_PM/%x_%j.out     # Output file
#SBATCH --error=/home/rgg2706/Multimodal-Sentiment-Analysis/Logs/ESAFN/015_Aug-08-2024_02:09_PM/%x_%j.err       # Error file

# Load necessary environment
spack env activate default-ml-x86_64-24071101

# Run the main script
stdbuf -oL -eL bash /home/rgg2706/Multimodal-Sentiment-Analysis/Models/ESAFN/run.sh
