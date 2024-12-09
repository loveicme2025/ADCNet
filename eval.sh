#!/usr/bin/env bash

gpus=0

data_name=levir
net_G=MPDNet
split=test
project_name=CD_MPDNet_LEVIR_b16_lr0.01_train_val_300_linear
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


