#!/usr/bin/env python

import os
import argparse
import xml.etree.ElementTree as ET

def preprocess_jpg():
    test_fid = open('test.txt')

    lines = test_fid.readlines()
    n_img = len(lines)
    annotation_files = []
    filenames = []
    for line in lines:
        annotation_file = line.split(' ')[1]
        annotation_files.append(annotation_file.split('\n')[0])
        file_name = annotation_file.split('/')[-1].split('.')[0]
        filenames.append(file_name)


    class_names = []
    class_names_fid = open('class_list.txt')
    for class_name in class_names_fid.readlines():
        class_names.append(class_name.split('\n')[0])

    gt_fids = {}
    for class_name in class_names:
        gt_fids[class_name] = open('gt/' + class_name + '_test_gt.txt', 'w')

    for i in xrange(n_img):  # each image

        tree = ET.parse(annotation_files[i])
        root = tree.getroot()

        n_instances = {}
        bboxes = {}
        for child in root:
            if child.tag == 'object':
                bndbox = child.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                class_name = child.find('name').text
                if class_name not in bboxes:
                    bboxes[class_name] = []
                bboxes[class_name].append([xmin, ymin, xmax, ymax])
                if class_name not in n_instances:
                    n_instances[class_name] = 0
                n_instances[class_name] += 1

        for class_name in class_names:
            gt_fid = gt_fids[class_name]
            gt_fid.write('{}\n'.format(filenames[i]))
            if class_name in n_instances:
                gt_fid.write('{}\n'.format(n_instances[class_name]))
                for inst_id in range(n_instances[class_name]):
                    bbox = bboxes[class_name][inst_id]
                    gt_fid.write('{:.2f} {:.2f} {:.2f} {:.2f} {}\n'.format(
                        bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], 0))
            else:
                gt_fid.write('{}\n'.format(0))

    for class_name in class_names:
        gt_fids[class_name].close()

def main():
    """
    Main function.
    """
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_dir', help='voc root directory', dest='voc_dir',
                        default='../voc')
    parser.add_argument('--output_dir', help='output data directory', dest='output_dir',
                        default='../voc_cache')
    args = parser.parse_args()

    image_dir = os.path.join(args.voc_dir, 'JPEGImages')  # voc training image directory
    label_dir = os.path.join(args.voc_dir, 'Annotations')  # voc training label directory

    preprocess_jpg()


if __name__ == '__main__':
    main()
