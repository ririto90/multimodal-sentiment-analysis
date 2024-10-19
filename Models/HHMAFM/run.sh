#!/usr/bin/env bash
for d in 'mvsa-mts' # 'mvsa-m' 'mvsa-m-100'
do
    for k in 'cmhafusion' # 'mmfusion' 'cmhafusion' 'mfcchfusion'
    do
        PYTHONPATH=$PYTHONPATH:/home/rgg2706/Multimodal-Sentiment-Analysis/Models/HHMAFM/src/ \
        CUDA_VISIBLE_DEVICES=0 python -u -Wd Models/HHMAFM/src/instructor_tests/train2_test.py --dataset ${d} --model_name ${k} \
        --num_epoch 10 --batch_size 64 --log_step 10
    done
done