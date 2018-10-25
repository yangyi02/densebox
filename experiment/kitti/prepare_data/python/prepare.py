#!/usr/bin/env python
"""
Load and parse kitti's raw label to Densebox reading format.

Usage
    ./prepare_kitti.py --kitti_dir kitti_dir --output_dir output_dir
    where
    kitti_dir is the path to the raw kitti dataset
    which should has the following structure:
        kitti_dir/training
        kitti_dir/training/image_2
        kitti_dir/training/label_2
        kitti_dir/testing
        kitti_dir/cpp
        kitti_dir/matlab

Example
    ./prepare_kitti.py --kitti_dir ../kitti --output_dir ../kitti_cache

History
    create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
    modify  -  Yi Yang (yangyi02@gmail.com), 2016-05
"""
import os
import argparse
from easydict import EasyDict
import numpy as np
import skimage.io
import random


def get_kitti_train(label_dir):
    """
    Given the directory of raw kitti labels, output an array of dictionaries,
    where each dictionary contain the information of an image.

    Input
        label_dir   -  directory for raw kitti labels

    Output
        annos     -  a list of dictionary
            type        -  element, i.e. 'Pedestrian'
            truncation  -  element, i.e. 0
            occlusion   -  element, i.e. 0
            alpha       -  element, i.e. -0.2000
            x1          -  element, i.e. 712.4000
            y1          -  element, i.e. 143
            x2          -  element, i.e. 810.7300
            y2          -  element, i.e. 307.9200
            h           -  element, i.e. 1.8900
            w           -  element, i.e. 0.4800
            l           -  element, i.e. 1.2000
            t           -  element, i.e. [1.8400 1.4700 8.4100]
            ry          -  element, i.e. 0.0100
    """
    # get total number of images
    n_img = len(os.listdir(label_dir))
    assert n_img == 7481, 'Kitti car detection dataset should contain totally 7481 training images'

    annos = []
    for i in xrange(n_img):
        label_path = os.path.join(label_dir, '{:06d}.txt'.format(i))
        with open(label_path) as fio:
            lines = fio.read().splitlines()

        num_line = len(lines)
        anno_i = []
        for line_id in xrange(num_line):
            line = lines[line_id]
            terms = line.split()
            assert len(terms) == 15

            anno = EasyDict()
            anno.type = terms[0]
            anno.truncation = float(terms[1])
            anno.occlusion = int(terms[2])
            anno.alpha = float(terms[3])
            anno.x1 = float(terms[4])
            anno.y1 = float(terms[5])
            anno.x2 = float(terms[6])
            anno.y2 = float(terms[7])
            anno.h = float(terms[8])
            anno.w = float(terms[9])
            anno.l = float(terms[10])
            anno.t = [float(terms[11 + j]) for j in xrange(3)]
            anno.ry = float(terms[14])
            anno_i.append(anno)

        annos.append(anno_i)

    return annos


def reformat_annotation(annos):
    """
    Reformat annotations for future usage.

    Input
      annos -  a 7481-length dictionary array containing kitti annotations

    Output
      output_annos  -  the new formatted output
    """

    # dimension
    n_img = len(annos)

    output_annos = EasyDict()
    output_annos.images = []
    output_annos.label_id = {}
    output_annos.label_name = []
    output_annos.label_count = []

    for i in xrange(n_img):  # each image
        if i % 500 == 0:  # show progress
            print '{}/{}'.format(i, n_img)

        # create a node
        image = EasyDict()
        image.name = '{:06d}'.format(i)
        image.instance = []
        image.bbox = []
        image.occlusion = []
        image.ignore = []
        image.label_id = []
        image.truncation = []

        # each instance labeled for each image
        cur_instances = annos[i]
        num_instance = len(annos[i])
        for instance_id in xrange(num_instance):
            instance = cur_instances[instance_id]

            # add bbox
            bbox = [0] * 4
            bbox[0] = instance.x1
            bbox[1] = instance.y1
            bbox[2] = instance.x2
            bbox[3] = instance.y2
            image.bbox.append(bbox)

            # add center
            point = [0] * 2
            point[0] = (bbox[0] + bbox[2]) / 2
            point[1] = (bbox[1] + bbox[3]) / 2
            image.instance.append(point)

            # add other information
            image.occlusion.append(instance.occlusion)
            image.truncation.append(instance.truncation)
            image.ignore.append(0)

            # for statics
            if instance.type in output_annos.label_id:
                output_annos.label_count[output_annos.label_id[instance.type]] += 1
            else:
                output_annos.label_id[instance.type] = len(output_annos.label_count)
                output_annos.label_name.append(instance.type)
                output_annos.label_count.append(1)
            image.label_id.append(output_annos.label_id[instance.type])

        output_annos.images.append(image)

    return output_annos


def filter_annotation(annos, min_height=25, max_occlusion=2, max_truncation=0.5, ignore_van=True):
    """
    Filter out annotations that do not match training criteria
    """
    output_annos = EasyDict()
    output_annos.label_name = annos.label_name
    output_annos.label_count = annos.label_count
    output_annos.images = []
    all_label_name = output_annos.label_name

    n_img = len(annos.images)
    for img_id in xrange(n_img):  # each image
        img_anno = annos.images[img_id]

        out_anno = EasyDict()
        out_anno.name = img_anno.name
        out_anno.instance = []
        out_anno.bbox = []
        out_anno.ignore = []
        out_anno.label_id = []
        out_anno.occlusion = []
        out_anno.truncation = []
        ignore_flag = [0] * len(img_anno.instance)

        for i in xrange(len(img_anno.instance)):
            if img_anno.ignore[i] == 1:
                ignore_flag[i] = 1
                continue

            cur_label_id = img_anno.label_id[i]
            cur_label_name = all_label_name[cur_label_id]
            if ignore_van:
                if cur_label_name == 'Van':
                    img_anno.ignore[i] = 1
                    continue

            if cur_label_name == 'DontCare':
                ignore_flag[i] = 1
                img_anno.ignore[i] = 1
                continue

            if cur_label_name != 'Van' and cur_label_name != 'Car' and cur_label_name != 'DontCare':
                ignore_flag[i] = 1
                continue

            bbox = img_anno.bbox[i]
            height = bbox[3] - bbox[1]

            if height < min_height or img_anno.occlusion[i] > max_occlusion \
                    or img_anno.truncation[i] > max_truncation:
                img_anno.ignore[i] = 1
                continue

        for i in xrange(len(img_anno.instance)):
            if ignore_flag[i] == 0:
                out_anno.instance.append(img_anno.instance[i])
                out_anno.bbox.append(img_anno.bbox[i])

                out_anno.ignore.append(img_anno.ignore[i])
                out_anno.label_id.append(img_anno.label_id[i])

                out_anno.occlusion.append(img_anno.occlusion[i])
                out_anno.truncation.append(img_anno.truncation[i])

        output_annos.images.append(out_anno)

    return output_annos


def preprocess_jpg(anno, image_dir, output_label_dir, output_image_dir):
    """
    Create final jpeg images and ground truth labels for DenseBox to read.
    """
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # since kitti does not contain validaiton set,
    # we use 95% training images for training and 5% training images for validation
    n_img = len(anno.images)
    n_val_img = int(round(n_img * 0.05))
    img_indices = range(n_img)
    random.shuffle(img_indices)
    val_img_indices = img_indices[:n_val_img]
    is_val = [False] * n_img
    for i in val_img_indices:
        is_val[i] = True

    # file handler
    train_fid = open(os.path.join(output_label_dir, 'new_original_train_jpg.txt'), 'w')
    val_fid = open(os.path.join(output_label_dir, 'new_original_val_jpg.txt'), 'w')
    val_filename_fid = open(os.path.join(output_label_dir, 'val_filename.txt'), 'w')
    gt_val_fid = open(os.path.join(output_label_dir, 'new_original_val_gt.txt'), 'w')

    for i in xrange(n_img):  # each image
        if i % 100 == 0:  # show progress
            print '{}/{}'.format(i, n_img)

        # original image
        full_file_name = os.path.join(image_dir, anno.images[i].name + '.png')
        file_name = anno.images[i].name.split('.')[0]
        n_instance = len(anno.images[i].instance)

        # load image
        img = skimage.io.imread(full_file_name)
        img_size_h, img_size_w, num_channel = img.shape

        # load instances
        ignore = anno.images[i].ignore
        bboxes = anno.images[i].bbox

        if is_val[i]:  # used for validation
            val_filename_fid.write(' ' + file_name + '\n')
            gt_val_fid.write('{}\n'.format(file_name))
            gt_val_fid.write('{}\n'.format(n_instance))
            for inst_id in range(n_instance):
                bbox = bboxes[inst_id]
                gt_val_fid.write('{:.2f} {:.2f} {:.2f} {:.2f} {}\n'.format(
                    bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], ignore[inst_id]))

        for is_flip in [False, True]:
            # flip image
            if is_flip:
                for channel_id in xrange(num_channel):
                    img[:, :, channel_id] = np.fliplr(img[:, :, channel_id])

            # write image
            if is_flip:
                skimage.io.imsave(os.path.join(output_image_dir, file_name + '_lr.jpg'), img)
            else:
                skimage.io.imsave(os.path.join(output_image_dir, file_name + '.jpg'), img)

            # write file name
            if is_val[i] == 0:
                cur_fid = train_fid
            else:
                cur_fid = val_fid
            if is_flip:
                cur_fid.write(file_name + '_lr')
            else:
                cur_fid.write(file_name)

            # write each instance
            for inst_id in xrange(n_instance):
                bbox = bboxes[inst_id]
                if is_flip:
                    bbox = [img_size_w - bbox[0], bbox[1], img_size_w - bbox[2], bbox[3]]
                    bbox = [bbox[2], bbox[1], bbox[0], bbox[3]]
                bboxes[inst_id] = bbox

                cur_fid.write(' {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} '.format(
                    bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3],
                    bbox[0] / 2 + bbox[2] / 2, bbox[1] / 2 + bbox[3] / 2))
                cur_fid.write('{:.2f} {:.2f} '.format(ignore[inst_id], ignore[inst_id]))
            cur_fid.write('\n')

    train_fid.close()
    val_fid.close()
    val_filename_fid.close()
    gt_val_fid.close()


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
    anno_list = get_kitti_train(label_dir)

    # transfer annotations to new annotation format
    annos = reformat_annotation(anno_list)

    # filter the ground truth annotations that does not match our criteria
    annos = filter_annotation(annos)

    # create final jpeg images and ground truth labels for DenseBox to read
    output_label_dir = os.path.join(args.output_dir, './annotations')
    output_image_dir = os.path.join(args.output_dir, './train_jpg')
    preprocess_jpg(annos, image_dir, output_label_dir, output_image_dir)


if __name__ == '__main__':
    main()
