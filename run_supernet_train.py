# All rights reserved.
"""
A script to run multinode training with submitit.
"""

import hf_env
hf_env.set_env('202105')

import os
import sys

def main():
    print(os.environ)
    node_rank = os.environ['RANK']
    world_size = os.environ['WORLD_SIZE']
    master_ip = os.environ['MASTER_IP']
    master_port = os.environ['MASTER_PORT']
    
    cmd = "python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=%s  \
    --node_rank=%s --master_addr=%s --master_port=%s --use_env supernet_train.py \
    --gp --change_qk --relative_position --mode super --dist-eval --load-pretrained-model \
    --cfg ./experiments/supernet/supernet-T.yaml --cfg-new  ./experiments/supernet/supernet-T-new.yaml\
    --epochs 500 --warmup-epochs 5 --lr 1e-3 --super_epoch 1 --step-num 7\
    --model deit_tiny_patch16_224 --batch-size 256 --resume ./result/supernet_tiny.pth\
    --output_dir ./result" % (world_size, node_rank, master_ip, master_port)

    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    main()
    