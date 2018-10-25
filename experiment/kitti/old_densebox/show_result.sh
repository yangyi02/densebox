#!/bin/sh

model="sgnet03"
gpu_id=1
cpu_or_gpu="GPU"
net_name="sgnet03_iter_597000.caffemodel"
output_dir="cache"
threshold="0.48"

export PIC_PRINT=0

cmd="../../caffe_densebox/build/tools/show_output.bin ./show_multiscale-sgnet-03.prototxt ./snapshot/${model}/${net_name} ${output_dir} ${cpu_or_gpu} ${gpu_id} ${threshold}"
echo $cmd

${cmd}
