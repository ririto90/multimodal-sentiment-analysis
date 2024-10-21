#!/usr/bin/env bash
for d in 'mvsa-mts' # 'mvsa-m' 'mvsa-m-100' 'mvsa-m-1000'
do
    for k in 'mfcchfusion2' # 'mmfusion' 'cmhafusion' 'mfcchfusion' 'mfcchfusion2'
    do
        cd ${REPO_DIR}
        echo "SLURM Job ID: $SLURM_JOB_ID"
        PYTHONPATH=$PYTHONPATH:/home/rgg2706/Multimodal-Sentiment-Analysis/Models/HHMAFM/src/ \
        python -u -Wd Models/HHMAFM/src/train.py --dataset ${d} --model_name ${k} \
        --num_epoch 10 --batch_size 32 --log_step 60
    done
done