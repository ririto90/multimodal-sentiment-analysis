#!/usr/bin/env bash
for d in 'mvsa-mts-1000' # 'mvsa-m' 'mvsa-m-100' 'mvsa-m-1000'
do
    for k in 'mmfusion' # 'mmfusion' 'cmhafusion' 'mfcchfusion' 'mfcchfusion2'
    do
        cd ${REPO_DIR}
        PYTHONPATH=$PYTHONPATH:/home/rgg2706/Multimodal-Sentiment-Analysis/Models/HHMAFM/src/ \
        python -u -Wd Models/HHMAFM/src/train.py --dataset ${d} --model_name ${k} \
        --num_epoch 10 --batch_size 128 --log_step 20
    done
done