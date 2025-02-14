#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
echo "$SCRIPT_DIR"

REPO_DIR=$(dirname "$SCRIPT_DIR")

# Check if MODEL_NAME is set to "default" or not set
if [ "$MODEL_NAME" = "default" ] || [ -z "$MODEL_NAME" ]; then
    echo "Error: Change MODEL_NAME variable"
    exit 1
fi

# Get current date and time
CURRENT_DATE=$(date +"%Y-%m-%d")
CURRENT_DATE_TIME=$(date +"%b-%d-%Y_%I:%M_%p")

# Variables
LOGS_DIR="${REPO_DIR}/Logs"
MODEL_LOGS_DIR="${LOGS_DIR}/${MODEL_NAME}/${CURRENT_DATE}"

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

# Print the current date and time for debugging
echo "CURRENT_DATE_TIME: $CURRENT_DATE + $CURRENT_TIME"

# Create new subfolder with the incremented ID and current date and time
NEW_LOG_DIR="${MODEL_LOGS_DIR}/$(printf "%03d" ${NEXT_ID})_${CURRENT_DATE_TIME}"
mkdir -p ${NEW_LOG_DIR}

# Print the new log directory for debugging
echo "NEW_LOG_DIR: $NEW_LOG_DIR"

# Set output and error paths
OUTPUT_PATH="${NEW_LOG_DIR}/output.log"
ERROR_PATH="${NEW_LOG_DIR}/error.log"

# Print the output and error paths for debugging
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "ERROR_PATH: $ERROR_PATH"

# Change to the model directory
cd ${REPO_DIR}/Models/${MODEL_NAME}/

# Run the main script and redirect output and error to the log files
bash run.sh > ${OUTPUT_PATH} 2> ${ERROR_PATH} &
