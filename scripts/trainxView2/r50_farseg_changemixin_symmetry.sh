#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

config_path='trainxView2.r50_farseg_changemixin_symmetry'
model_dir='./log/changestar_sisup/r50_farseg_changemixin_symmetry'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 6699 ./train_changemixin.py \
  --config_path=${config_path} \
  --model_dir=${model_dir}
