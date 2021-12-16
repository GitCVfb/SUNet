#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_training_data=/home1/fanbin/fan/raw_data/carla/data_train/train/
                          
fastec_dataset_type=Fastec
fastec_root_path_training_data=/home1/fanbin/fan/raw_data/faster/data_train/train/

log_dir=/home1/fanbin/fan/SUNet/deep_unroll_weights/
#
cd deep_unroll_net

python train_symmetry.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_training_data \
          --log_dir=$log_dir \
          --lamda_perceptual=1 \
          --lamda_L1=10 \
          --lamda_flow_smoothness=0.1 \
          --crop_sz_H=448 \
          #--continue_train=True \
          #--start_epoch=111 \
          #--model_label=110 \

#python train_symmetry.py \
#          --dataset_type=$fastec_dataset_type \
#          --dataset_root_dir=$fastec_root_path_training_data \
#          --log_dir=$log_dir \
#          --lamda_perceptual=1 \
#          --lamda_L1=10 \
#          --lamda_flow_smoothness=0.1 \
#          --crop_sz_H=480 \
          #--continue_train=True \
          #--start_epoch=111 \
          #--model_label=110

