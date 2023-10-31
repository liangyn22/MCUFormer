# MCUFormer: Deploying Vision Transformers on Microcontrollers with Limited Memory

**This is an official implementation of AutoFormer.**

MCUFormer is an one-shot network architecture search (NAS) to discover the optimal architecture with highest task performance given the memory budget from the microcontrollers. For the construction of the inference operator library of vision transformers, we schedule the memory buffer during inference through operator integration, patch embedding decomposition, and token overwriting, allowing the memory buffer to be fully utilized to adapt to the forward pass of the vision transformer.

## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
conda create -n MCUFormer python=3.7
conda activate MCUFormer
pip install -r requirements.txt
```

## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>

The directory structure is the standard layout as following.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Quick Start
We provide *Supernet Train, Search, Test* code of AutoFormer as follows.

### Supernet Train 

You can run the following command to get the optimal supernet.
```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --mode super --dist-eval --load-pretrained-model \
--cfg ./experiments/supernet/supernet-T.yaml --cfg-new  ./experiments/supernet/supernet-T-new.yaml \
--epochs 500 --warmup-epochs 5 --lr 1e-4 --super_epoch 1 --step-num 7 \
--model deit_tiny_patch16_224 --batch-size 128 --output /OUTPUT_PATH
```

### Search
We run our evolution search on part of the ImageNet training dataset and use the validation set of ImageNet as the test set for fair comparison. To generate the subImagenet in `/PATH/TO/IMAGENET`, you could simply run:
```buildoutcfg
python ./lib/subImageNet.py --data-path /PATH/TO/IMAGENT
```

After obtaining the subImageNet and training of the supernet. You can run the following command to search the optimal subnet. Please remember to config the specific constraint in this evolution search using `--memory-constraint`: 
```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --dist-eval --input-size 240 --resume /PATH/TO/CHECKPOINT \
--cfg ./experiments/supernet/supernet-T.yaml --cfg-new  ./experiments/supernet/supernet-T-new.yaml \
--memory-constraint YOUR/CONFIG  --data-set EVO_IMNET --output_dir ./result/evolution_0.9_20 /OUTPUT_PATH 
```

## Todo List
We are fixing the code of detection and we will realease the enging code in few days.