#!/usr/bin/env bash

PORT=${PORT:-25410}

CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/lyn/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=$PORT \
    $(dirname "$0")/evolution.py  \
    --gp --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml  \
    --resume ./result/supernet/checkpoint.pth \
    --min-param-limits 4 --param-limits 8 --data-set EVO_IMNET \
    --batch-size 512 --input-size 240 --patch_size 20 \
    --output_dir ./result/evolution \
    --min-lr 1e-6 --lr 25e-6 

# --finetune ./models/deit_240/deit_tiny_patch20_240_weight.pth \
# --min-lr 1e-6 --lr 5e-5 \
