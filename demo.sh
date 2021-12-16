#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_fastec
mkdir -p experiments/results_demo_carla

cd deep_unroll_net


python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla \
            --data_dir='../demo/Carla' \
            --crop_sz_H=448 \
            --log_dir=../deep_unroll_weights/carla

python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_fastec \
            --data_dir='../demo/Fastec' \
            --crop_sz_H=480 \
            --log_dir=../deep_unroll_weights/fastec
