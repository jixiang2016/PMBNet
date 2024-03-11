#!/usr/bin/env bash

### data: [GOPRO-VFI_copy, RD_VFI]  
### mode: [RS, RSGR, Blur]
### Please specify your 'input_dir', 'model_dir' and etc.
CUDA_VISIBLE_DEVICES=0 python3 test.py \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='RD_VFI'  --data_mode='RS' \
		--output_dir='./output' \
		--model_dir='./train_log/RD_VFI_RS/checkpoint.ckpt' --batch_size=8 --keep_frames 