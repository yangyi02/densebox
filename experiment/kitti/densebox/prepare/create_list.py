#!/usr/bin/env python

import os
import argparse
import random


def main():
    """
    Main function.
    """
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dir', help='kitti root directory', dest='kitti_dir',
                        default='../kitti')
    parser.add_argument('--output_dir', help='output data directory', dest='output_dir',
                        default='../kitti_cache')
    args = parser.parse_args()

    image_dir = os.path.join(args.kitti_dir, 'training/image_2')  # kitti training image directory
    label_dir = os.path.join(args.kitti_dir, 'training/label_2')  # kitti training label directory

    # load training annotations from kitti dataset
    # get total number of images
    labels = os.listdir(label_dir)
    n_img = len(labels)
    assert n_img == 7481, 'Kitti car detection dataset should contain totally 7481 training images'

    # since kitti does not contain validaiton set,
    # we use 95% training images for training and 5% training images for validation
    n_test_img = int(round(n_img * 0.05))
    random.shuffle(labels)
    test_labels = labels[:n_test_img]
    train_labels = labels[n_test_img:]

    output_train_list_fid = open(os.path.join(args.output_dir, 'trainval.txt'), 'w')
    output_test_list_fid = open(os.path.join(args.output_dir, 'test.txt'), 'w')

    to_write_format = 'image_2/{} label_2/{}\n'
    for label in train_labels:
        file = label.split('.')[0] + '.png'
        output_train_list_fid.write(to_write_format.format(file, label))

    for label in test_labels:
        file = label.split('.')[0] + '.png'
        output_test_list_fid.write(to_write_format.format(file, label))

if __name__ == '__main__':
    main()
