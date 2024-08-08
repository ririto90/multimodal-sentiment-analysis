#!/usr/bin/env bash
for d in 'mvsa-m' # 'twitter2015' 'twitter2017' 'mvsa-m'
do
    for k in 'mmfusion' # 'mmram' 'mmmgan' 'mmfusion'
    do
        CUDA_VISIBLE_DEVICES=0 python -u train.py --dataset ${d} --embed_dim 100 --hidden_dim 100 --model_name ${k} \
        --att_mode vis_concat_attimg_gate --batch_size 128 --num_epoch 100
        # test --load_check_point >> ${d}_log.txt
    done
done
