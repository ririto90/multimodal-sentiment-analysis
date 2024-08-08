#!/bin/bash

REPO_DIR="${HOME}/Multimodal-Sentiment-Analysis"

# Check if MODEL_NAME is set to "default" or not set
if [ "$MODEL_NAME" = "default" ] || [ -z "$MODEL_NAME" ]; then
    echo "Error: Change MODEL_NAME variable"
    exit 1
fi

# Variables
LOGS_DIR="${REPO_DIR}/Logs"
MODEL_LOGS_DIR="${LOGS_DIR}/${MODEL_NAME}"

# Print the logs directory for debugging
echo "LOGS_DIR: $LOGS_DIR"
echo "MODEL_LOGS_DIR: $MODEL_LOGS_DIR"

# Create model logs directory if it doesn't exist
mkdir -p ${MODEL_LOGS_DIR}

# List contents of the logs directory for debugging
echo "Contents of MODEL_LOGS_DIR:"
ls -1v ${MODEL_LOGS_DIR}

# Get the next ID for the subfolder
NEXT_ID=$(ls -1v ${MODEL_LOGS_DIR} | grep -Eo '^[0-9]+' | sort -V | tail -n 1)

# Print the extracted NEXT_ID for debugging
echo "Extracted NEXT_ID: $NEXT_ID"

# Strip leading zeros
NEXT_ID=$(echo $NEXT_ID | sed 's/^0*//')

if [ -z "$NEXT_ID" ]; then
    NEXT_ID=1
else
    NEXT_ID=$((NEXT_ID+1))
fi

# Print the incremented NEXT_ID for debugging
echo "Incremented NEXT_ID: $NEXT_ID"

# Get current date and time
CURRENT_DATE_TIME=$(date +"%b-%d-%Y_%I:%M_%p")

# Print the current date and time for debugging
echo "CURRENT_DATE_TIME: $CURRENT_DATE_TIME"

# Create new subfolder with the incremented ID and current date and time
NEW_LOG_DIR="${MODEL_LOGS_DIR}/$(printf "%03d" ${NEXT_ID})_${CURRENT_DATE_TIME}"
mkdir -p ${NEW_LOG_DIR}

# Print the new log directory for debugging
echo "NEW_LOG_DIR: $NEW_LOG_DIR"

# Set output and error paths
OUTPUT_PATH="${NEW_LOG_DIR}/%x_%j.out"
ERROR_PATH="${NEW_LOG_DIR}/%x_%j.err"

# Print the output and error paths for debugging
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "ERROR_PATH: $ERROR_PATH"

# Define a new location for the temporary Slurm script
TEMP_SCRIPT_DIR="${REPO_DIR}/TempScripts"
mkdir -p ${TEMP_SCRIPT_DIR}  # Create the directory if it doesn't exist

# Print the temporary script directory for debugging
echo "TEMP_SCRIPT_DIR: $TEMP_SCRIPT_DIR"

# Set the temporary Slurm script path
TEMP_SLURM_SCRIPT="${NEW_LOG_DIR}/${MODEL_NAME}_script.sh"

# Print the temporary Slurm script path for debugging
echo "TEMP_SLURM_SCRIPT: $TEMP_SLURM_SCRIPT"

# Create the temporary Slurm script with substituted paths
cat <<EOT > ${TEMP_SLURM_SCRIPT}
#!/bin/bash -l

#SBATCH --job-name=${MODEL_NAME}    # Name of your job
#SBATCH --account=multisass         # Your Slurm account
#SBATCH --partition=tier3           # Run on tier3
#SBATCH --time=0-04:00:00           # 4 hours time limit
#SBATCH --nodes=1                   # # of nodes
#SBATCH --ntasks=1                  # 1 task (i.e. process)
#SBATCH --mem=32g                   # Increase RAM to 16GB
#SBATCH --gres=gpu:a100:2           # 1 a100 GPU
#SBATCH --output=${OUTPUT_PATH}     # Output file
#SBATCH --error=${ERROR_PATH}       # Error file

# Load necessary environment
spack env activate default-ml-x86_64-24071101

# Run the main script
cd ${REPO_DIR}/Models/${MODEL_NAME}/
stdbuf -oL -eL bash ${REPO_DIR}/Models/${MODEL_NAME}/run.sh
EOT

# Print the contents of the temporary Slurm script
echo "Temporary Slurm script content:"
cat ${TEMP_SLURM_SCRIPT}

# Make the temporary script executable
chmod +x ${TEMP_SLURM_SCRIPT}

# Print the command to submit the script for debugging
echo "Submitting script with command: sbatch ${TEMP_SLURM_SCRIPT}"

# Submit the temporary Slurm script
sbatch ${TEMP_SLURM_SCRIPT}
