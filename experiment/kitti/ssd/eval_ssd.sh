#!/bin/bash

dataset="KITTI"
iterations="10000"
input_size="552x552"
GPU_ID="2"
base_model="Inception"

echo ${base_model}
caffe train \
--solver=models/${base_model}Net/${dataset}/SSD_${input_size}/eval_solver.prototxt \
--weights=models/${base_model}Net/${dataset}/SSD_${input_size}/${base_model}_${dataset}_SSD_${input_size}_iter_${iterations}.caffemodel \
--gpu=${GPU_ID}

