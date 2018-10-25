#!/bin/sh

class="VOC"
GPU_id="3"
CPUorGPU="GPU"
NetName="VOC/multiscale-vgg-16_iter_20000.caffemodel"
outputFld="TestResultVOC2007"
show_result="1"
out_name="1000_voc_vgg_16.txt"
show_forward_time="1"

# export PIC_PRINT=1
#export SHOW_TIME=1


mode_str="${class}/multiscale-vgg-16"
echo $mode_str

cmd="pyramid_test test_multi-scale-vgg-16.prototxt snapshot/${NetName}  cache/${outputFld}  ${CPUorGPU} ${GPU_id}  ${show_result} ${out_name} ${show_forward_time}"
echo $cmd

${cmd}
