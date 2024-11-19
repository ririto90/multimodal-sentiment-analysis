#!/bin/bash

REPO_DIR="${HOME}/Multimodal-Sentiment-Analysis"

# Check if MODEL_NAME is set to "default" or not set
if [ "$MODEL_NAME" = "default" ] || [ -z "$MODEL_NAME" ]; then
    echo "> Error: Change MODEL_NAME variable"
    exit 1
fi

# Get current date and time
CURRENT_DATE=$(date +"%Y-%m-%d")
CURRENT_DATE_TIME=$(date +"%b-%d-%Y_%I:%M_%p")

# Variables
LOGS_DIR="${REPO_DIR}/Logs"
MODEL_LOGS_DIR="${LOGS_DIR}/${MODEL_NAME}"

# Initialize CURRENT_DATE_FOLDER
CURRENT_DATE_FOLDER="${CURRENT_DATE}"

# If increment variable is set to true, modify the date folder name
if [ "$increment" = "true" ]; then
    # Find the next sub_N number
    SUB_NEXT_ID=$(ls -1v ${MODEL_LOGS_DIR} 2>/dev/null | grep -E "^${CURRENT_DATE}-sub_[0-9]+" | sed "s/^${CURRENT_DATE}-sub_//" | sort -V | tail -n 1)
    # Strip leading zeros
    SUB_NEXT_ID=$(echo $SUB_NEXT_ID | sed 's/^0*//')
    if [ -z "$SUB_NEXT_ID" ]; then
        SUB_NEXT_ID=1
    else
        SUB_NEXT_ID=$((SUB_NEXT_ID+1))
    fi
    # Update CURRENT_DATE_FOLDER to include the sub_N
    CURRENT_DATE_FOLDER="${CURRENT_DATE}-sub_${SUB_NEXT_ID}"
else
    # Check if there are any subfolders with -sub_N
    EXISTING_SUB_FOLDERS=$(ls -1v ${MODEL_LOGS_DIR} 2>/dev/null | grep -E "^${CURRENT_DATE}-sub_[0-9]+")
    if [ -n "$EXISTING_SUB_FOLDERS" ]; then
        # Find the highest N
        SUB_NEXT_ID=$(echo "$EXISTING_SUB_FOLDERS" | sed "s/^${CURRENT_DATE}-sub_//" | sed 's/^0*//' | sort -V | tail -n 1)
        # Update CURRENT_DATE_FOLDER to include the highest sub_N
        CURRENT_DATE_FOLDER="${CURRENT_DATE}-sub_${SUB_NEXT_ID}"
    else
        # Use the standard date folder
        CURRENT_DATE_FOLDER="${CURRENT_DATE}"
    fi
fi

# Update MODEL_LOGS_DIR to include the date (modified if increment is true or existing subfolders are found)
MODEL_LOGS_DIR="${MODEL_LOGS_DIR}/${CURRENT_DATE_FOLDER}"

# Create model logs directory if it doesn't exist
mkdir -p ${MODEL_LOGS_DIR}

# Get the next ID for the subfolder
NEXT_ID=$(ls -1v ${MODEL_LOGS_DIR} | grep -Eo '^[0-9]+' | sort -V | tail -n 1)

# Strip leading zeros
NEXT_ID=$(echo $NEXT_ID | sed 's/^0*//')

if [ -z "$NEXT_ID" ]; then
    NEXT_ID=1
else
    NEXT_ID=$((NEXT_ID+1))
fi

# Create new subfolder with the incremented ID and current date and time
NEW_LOG_DIR="${MODEL_LOGS_DIR}/$(printf "%03d" ${NEXT_ID})_${CURRENT_DATE_TIME}"
mkdir -p ${NEW_LOG_DIR}

# Print the new log directory for debugging
echo "> NEW_LOG_DIR: $NEW_LOG_DIR"

# Set output and error paths
OUTPUT_PATH="${NEW_LOG_DIR}/%x_%j.out"
ERROR_PATH="${NEW_LOG_DIR}/%x_%j.err"

# Define a new location for the temporary Slurm script
TEMP_SCRIPT_DIR="${REPO_DIR}/TempScripts"
mkdir -p ${TEMP_SCRIPT_DIR}  # Create the directory if it doesn't exist


# Set the temporary Slurm script path
TEMP_SLURM_SCRIPT="${NEW_LOG_DIR}/${MODEL_NAME}_script.sh"

# Create the temporary Slurm script with substituted paths
cat <<EOT > ${TEMP_SLURM_SCRIPT}
#!/bin/bash -l

#SBATCH --job-name=${MODEL_NAME}    # Name of your job
#SBATCH --account=multisass         # Your Slurm account
#SBATCH --partition=tier3           # Run on tier3
#SBATCH --time=0-12:00:00           # 12 hours time limit
#SBATCH --nodes=1                   # # of nodes
#SBATCH --ntasks=1                  # 1 task (i.e. process)
#SBATCH --mem=32g                   # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:2           # 1 a100 GPU
#SBATCH --output=${OUTPUT_PATH}     # Output file
#SBATCH --error=${ERROR_PATH}       # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variable (explicitly within the script) for access within the python script environment
export NEW_LOGS_DIR="${NEW_LOG_DIR}"

# Run the main script
cd ${REPO_DIR}/

stdbuf -oL -eL bash ${REPO_DIR}/Models/${MODEL_NAME}/run.sh
EOT

# Make the temporary script executable
chmod +x ${TEMP_SLURM_SCRIPT}

# Submit the temporary Slurm script
sbatch ${TEMP_SLURM_SCRIPT}
