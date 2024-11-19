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

# Create date folder if it doesn't exist
DATE_FOLDER="${MODEL_LOGS_DIR}/${CURRENT_DATE}"
mkdir -p "${DATE_FOLDER}"

# If sub_name variable is set, create a new sub-N folder with sub_name
if [ -n "$sub_name" ]; then
    # Find the next sub-N number (consider all subfolders starting with sub-)
    SUB_NEXT_ID=$(ls -1v "${DATE_FOLDER}" 2>/dev/null | grep -E "^sub-[0-9]+" | sed -E 's/^sub-([0-9]+).*/\1/' | sed 's/^0*//' | sort -n | tail -n 1)
    if [ -z "$SUB_NEXT_ID" ]; then
        SUB_NEXT_ID=1
    else
        SUB_NEXT_ID=$((SUB_NEXT_ID+1))
    fi
    # Sanitize sub_name for folder name
    CLEAN_SUB_NAME=$(echo "$sub_name" | sed 's/[^a-zA-Z0-9_-]/_/g')
    # Define SUB_FOLDER
    SUB_FOLDER="${DATE_FOLDER}/sub-${SUB_NEXT_ID}_${CLEAN_SUB_NAME}"
else
    # Check if there are any subfolders with exact name sub-N (no suffix)
    EXISTING_SUB_FOLDERS=$(ls -1v "${DATE_FOLDER}" 2>/dev/null | grep -E "^sub-[0-9]+$")
    if [ -n "$EXISTING_SUB_FOLDERS" ]; then
        # Find the highest N
        SUB_NEXT_ID=$(echo "$EXISTING_SUB_FOLDERS" | sed 's/^sub-//' | sed 's/^0*//' | sort -n | tail -n 1)
        SUB_FOLDER="${DATE_FOLDER}/sub-${SUB_NEXT_ID}"
    else
        # Use sub-1 as default
        SUB_FOLDER="${DATE_FOLDER}/sub-1"
    fi
fi

# Create the subfolder if it doesn't exist
mkdir -p "${SUB_FOLDER}"

# Now, under the SUB_FOLDER, we create the NEW_LOG_DIR
# Get the next ID for the subfolder
NEXT_ID=$(ls -1v "${SUB_FOLDER}" | grep -Eo '^[0-9]+' | sort -n | tail -n 1)
NEXT_ID=$(echo $NEXT_ID | sed 's/^0*//')

if [ -z "$NEXT_ID" ]; then
    NEXT_ID=1
else
    NEXT_ID=$((NEXT_ID+1))
fi

# Create new subfolder with the incremented ID and current date and time
NEW_LOG_DIR="${SUB_FOLDER}/$(printf "%03d" ${NEXT_ID})_${CURRENT_DATE_TIME}"
mkdir -p "${NEW_LOG_DIR}"

# Print the new log directory for debugging
echo "> NEW_LOG_DIR: $NEW_LOG_DIR"

# Set output and error paths
OUTPUT_PATH="${NEW_LOG_DIR}/%x_%j.out"
ERROR_PATH="${NEW_LOG_DIR}/%x_%j.err"

# Copy the run.sh script to the NEW_LOG_DIR
cp "${REPO_DIR}/Models/${MODEL_NAME}/run.sh" "${NEW_LOG_DIR}/run.sh"

# Set the temporary Slurm script path
TEMP_SLURM_SCRIPT="${NEW_LOG_DIR}/${MODEL_NAME}_script.sh"

# Create the temporary Slurm script with substituted paths
cat <<EOT > "${TEMP_SLURM_SCRIPT}"
#!/bin/bash -l

#SBATCH --job-name=${MODEL_NAME}    # Name of your job
#SBATCH --account=multisass         # Your Slurm account
#SBATCH --partition=tier3           # Run on tier3
#SBATCH --time=0-12:00:00           # 12 hours time limit
#SBATCH --nodes=1                   # # of nodes
#SBATCH --ntasks=1                  # 1 task (i.e. process)
#SBATCH --mem=32g                   # Increase RAM to 32GB
#SBATCH --gres=gpu:a100:2           # 2 a100 GPUs
#SBATCH --output=${OUTPUT_PATH}     # Output file
#SBATCH --error=${ERROR_PATH}       # Error file

# Load necessary environment
spack env activate default-nlp-x86_64-24072401

# Set the environment variables for access within the python script environment
export NEW_LOGS_DIR="${NEW_LOG_DIR}"
export REPO_DIR="${REPO_DIR}"

# Run the main script
cd "${NEW_LOG_DIR}"
echo "> NEW_LOG_DIR: \$NEW_LOG_DIR"
echo "Current directory: \$(pwd)"

stdbuf -oL -eL bash run.sh
EOT

# Make the temporary script executable
chmod +x "${TEMP_SLURM_SCRIPT}"

# Submit the temporary Slurm script
sbatch "${TEMP_SLURM_SCRIPT}"
