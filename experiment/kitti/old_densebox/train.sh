#!/bin/sh

snapshot="--snapshot=snapshot/sgnet03/sgnet03_iter_515000.solverstate"

model="sgnet03"
gpu_id=0,1,2,3

if [ ! -d "log" ]; then
    mkdir "log"
fi
if [ ! -d "snapshot" ]; then
    mkdir "snapshot"
fi
if [ ! -d "snapshot/${model}" ]; then
    mkdir "snapshot/${model}"
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

if [ -z ${snapshot} ]; then
    cmd="caffe train --solver=./multiscale-sgnet-03-solver.prototxt --gpu=${gpu_id} "
else
    cmd="caffe train --solver=./multiscale-sgnet-03-solver.prototxt ${snapshot} --gpu=${gpu_id} "
fi
echo $cmd

#${cmd} $1 2>&1 | tee ${tee_arg} log/${model}.log 
nohup ${cmd} 2>&1 1>log/${model}.log &
#nohup ${cmd} 2>&1 1>temp.log &

