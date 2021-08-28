#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

config_path='levircd.rx101_32x4d_farseg_changestar_bisup'
model_dir='./log/bisup-LEVIRCD/rx101_32x4d_farseg_changestar'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9558 ./train_sup_change.py \
  --config_path=${config_path} \
  --model_dir=${model_dir}