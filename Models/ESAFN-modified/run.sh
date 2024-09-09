#!/usr/bin/env bash
for d in 'mvsa-mts' # 'mvsa-m' 'mvsa-m-100'
do
    for k in 'mmfusion' # 'mmfusion'
    do
        CUDA_VISIBLE_DEVICES=0 python -u -Wd train.py --dataset ${d} --embed_dim 100 --hidden_dim 100 --model_name ${k} \
        --att_mode vis_concat_attimg_gate --batch_size 16 --log_step 32 --num_epoch 100
    done
done
