#!/usr/bin/env bash
# Generate the list and image for densebox.
# The intermediate preprocessed data would be put under ./cache folder.
#
# Usage
#   ./prepare_data.sh kitti_dir
# where kitti_dir is the kitti dataset root path
#
# Example
#  ./prepare_data.sh /media/yi/DATA/data-orig/kitti
python prepare_data/python/prepare.py --kitti_dir $1 --output_dir cache
