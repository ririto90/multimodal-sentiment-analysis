#!/bin/bash

# Model Variables
MODEL_NAME='DMLAN'
fusion='dmlanfusion' # 'dmlanfusion' 'dmlanfusion2'
dataset='mvsa-mts-v3' # 'mvsa-mts-v3' 'mvsa-mts-v3-1000'
lr='0.001'
dr='0.5'
batch_size='64'
epochs=5

# Slurm Variables
partition='tier3' # 'tier3'
memory='256' # '64' '128' '256'