# Preparing training and testing data for kitti car detection #

### Note ###

* Both matlab script and python script work for the same data preprocessing.
* Note that currently, the python script for preprocessing kitti car dataset is the latest.
* There are slightly differences between the python script and matlab script.

### Train / Validation Split ###
* Since there are no kitti validation set, we use 95% training images for training and 5% training images for validation.
* The training and validation set are randomly selected, so there may be small difference in terms of performance evaluation.
