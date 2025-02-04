#!/bin/bash

MODEL_NAME='DMLAN'

REPOSITORY='/Users/ronengold/Library/Mobile Documents/com~apple~CloudDocs/Repositories'

fusion='dmlanfusion2' # 'dmlan'
dataset='mvsa-mts-v3-30' # 'mvsa-mts-v3' 'mvsa-mts-v3-1000'
lr='0.001'
dr='0.5'


REPO_DIR="${REPOSITORY}/multimodal-sentiment-analysis-main"

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

# If increment variable is set to true, create a new sub_N folder
if [ "$increment" = "true" ]; then
    # Find the next sub_N number
    SUB_NEXT_ID=$(ls -1v "${DATE_FOLDER}" 2>/dev/null | grep -E "^sub-[0-9]+" | sed "s/^sub-//" | sed 's/^0*//' | sort -n | tail -n 1)
    if [ -z "$SUB_NEXT_ID" ]; then
        SUB_NEXT_ID=1
    else
        SUB_NEXT_ID=$((SUB_NEXT_ID+1))
    fi
    # Define SUB_FOLDER
    SUB_FOLDER="${DATE_FOLDER}/sub-${SUB_NEXT_ID}"
else
    # Check if there are any subfolders with sub_N
    EXISTING_SUB_FOLDERS=$(ls -1v "${DATE_FOLDER}" 2>/dev/null | grep -E "^sub-[0-9]+")
    if [ -n "$EXISTING_SUB_FOLDERS" ]; then
        # Find the highest N
        SUB_NEXT_ID=$(echo "$EXISTING_SUB_FOLDERS" | sed "s/^sub-//" | sed 's/^0*//' | sort -n | tail -n 1)
        SUB_FOLDER="${DATE_FOLDER}/sub-${SUB_NEXT_ID}"
    else
        # Use sub_1 as default
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
OUTPUT_PATH="${NEW_LOG_DIR}/output.log"
ERROR_PATH="${NEW_LOG_DIR}/error.log"

# Set the environment variables
export NEW_LOGS_DIR="${NEW_LOG_DIR}"
fusion="${fusion}"
dataset="${dataset}"
lr="${lr}"
dr="${dr}"

# Run the main script
cd "${REPO_DIR}"

PYTHONPATH="$PYTHONPATH:${REPO_DIR}" \
python -u -Wd "${REPO_DIR}/Models/${MODEL_NAME}/src/train.py" \
    --model_fusion "${fusion}" \
    --dataset "${dataset}" \
    --num_epoch 2 \
    --batch_size 64 \
    --log_step 60 \
    --learning_rate "${lr}" \
    --dropout_rate "${dr}" \
    --weight_decay 0 \
    > "${OUTPUT_PATH}" 2> "${ERROR_PATH}"