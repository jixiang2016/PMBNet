#!/usr/bin/env bash

### data： GOPRO-VFI_copy, RD_VFI
### data_mode: ['RSGR', 'RS','Blur']   
### The image height and width should be multiples of 32
### Before traing, please update "input_dir", "dataset_name" and etc.

CUDA_VISIBLE_DEVICES=0,1  python3 -m torch.distributed.launch --master_port=16645 --nproc_per_node=2 train.py \
        --world_size=2 \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='GOPRO-VFI_copy' --data_mode='RS' \
		--output_dir='./train_log' --batch_size_val=1 \
		--batch_size=4  --epoch=800 #--resume=True