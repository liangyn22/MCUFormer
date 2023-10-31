#!/usr/bin/env bash

PORT=${PORT:-25416}

CUDA_VISIBLE_DEVICES=0,1,2 /home/ubuntu/anaconda3/envs/lyn/bin/python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port=$PORT \
    $(dirname "$0")/supernet_train.py  \
    --gp --change_qk --relative_position --mode super --dist-eval --load-pretrained-model \
    --cfg ./experiments/supernet/supernet-T.yaml --cfg-new  ./experiments/supernet/supernet-T-new.yaml\
    --epochs 500 --warmup-epochs 5 --lr 1e-4 --super_epoch 1 --step-num 7\
    --model deit_tiny_patch16_224 --batch-size 256 --resume ./result/supernet_tiny.pth\
    --output_dir ./result \
    

# --finetune ./models/deit_240/deit_tiny_patch20_240_weight.pth \
# --min-lr 1e-6 --lr 5e-5 \
# --resume ./result/supernet/checkpoint.pth \