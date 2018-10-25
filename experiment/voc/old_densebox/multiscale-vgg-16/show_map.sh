#!/bin/sh

class="VOC"
GPU_id="0"
CPUorGPU="GPU"
NetName="VOC/multiscale-vgg-16-2_iter_66000.caffemodel"
outputFld="output"
threshold="0.1"

export PIC_PRINT=0

mode_str="${class}/multiscale-vgg-16-2"
echo $mode_str

cmd="./build/tools/show_output.bin examples/${mode_str}/show_multi-scale-vgg-16.prototxt snapshot/${NetName}  cache/${outputFld}  ${CPUorGPU} ${GPU_id} ${threshold}"
echo $cmd

make -j 8 && ${cmd}
