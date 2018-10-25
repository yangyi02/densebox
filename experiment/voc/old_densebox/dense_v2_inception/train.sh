#!/bin/sh

#snapshot="--snapshot=snapshot/VOC/multiscale-vgg-16_iter_24000.solverstate"
#weight="--weights=./VGG_ILSVRC_16_layers.caffemodel"

gpu_id=0,1,2,3
class='VOC'
mode_str="${class}/dense_v2_inception"
echo $mode_str
if [ ! -d "log" ]; then
mkdir "log"
fi

if [ ! -d "log/${class}" ]; then
mkdir "log/${class}"
fi

tee_arg=''
if [ $# == 1 ]; then
tee_arg='-a'
echo 'Restore from '$1
fi

#############################################
# the following variables are used for debug
# 1 for activation

# print the input label in ImageMultiScaleDataLayer/ImageDataLayer
export LABEL_PRINT=0

# print the input image
export PIC_PRINT=0

# print the mask in label_related_dropout layer
export LABEL_RELATED_DROPOUT_PRINT=0

# print the hard samples which are selected during bootstrap step
export BOOTSTRAP_PRINT=0

# redirect info to error(glog)
export GLOG_logtostderr=1

#############################################

cmd="caffe train --solver=dense_v2_inception-solver.prototxt ${snapshot} --gpu=${gpu_id}"
echo $cmd

#${cmd} $1 2>&1 | tee ${tee_arg} log/${mode_str}.log 
#nohup ${cmd} 2>&1 1>log/${model_str}.log &
nohup ${cmd} 2>&1 1>dense_v2_inception.log &


