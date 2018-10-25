import argparse
import os
import shutil
import subprocess
import sys

from caffe.proto import caffe_pb2
from google.protobuf import text_format

if __name__ == "__main__":

    dataset = sys.argv[1]
    caffe_root = '/home/vis/heming/caffe-ssd'
    root_dir = '/home/vis/heming/data/BaiduCar/Baidu_data_test/20151204/imgs/'
    list_file = '/home/vis/heming/data/BaiduCar/Baidu_data_test/20151204/Annotations/{}.txt'.format(dataset)
    out_dir = '/home/vis/heming/data/BaiduCar/lmdb/baidu_{}_lmdb'.format(dataset)
    example_dir = 'examples/Baidu'

    need_name_size = 1 #True
    name_size_file = 'data/Baidu/{}_name_size.txt'.format(dataset)

    redo = 1
    anno_type = "detection"
    backend = "lmdb"
    check_size = False
    encode_type = "jpg"
    encoded = "True"
    gray = "False"
    min_dim = 0
    max_dim = 0
    resize_height = 0
    resize_width = 0
    shuffle = False #True
    #check_label = True

    # check if root directory exists
    if not os.path.exists(root_dir):
        print "root directory: {} does not exist".format(root_dir)
        sys.exit()

    out_parent_dir = os.path.dirname(out_dir)
    if not os.path.exists(out_parent_dir):
        os.makedirs(out_parent_dir)
    if os.path.exists(out_dir) and not redo:
        print "{} already exists and I do not hear redo".format(out_dir)
        sys.exit()
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if anno_type == "detection":
        cmd = "{}/build/tools/convert_baidu_set" \
              " --min_dim={}" \
              " --max_dim={}" \
              " --resize_height={}" \
              " --resize_width={}" \
              " --backend={}" \
              " --shuffle={}" \
              " --check_size={}" \
              " --encode_type={}" \
              " --encoded={}" \
              " --gray={}" \
              " {} {} {}" \
            .format(caffe_root,
                    min_dim, max_dim, resize_height, resize_width, backend, shuffle,
                    check_size, encode_type, encoded, gray, root_dir, list_file, out_dir)

    print cmd
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

    if not os.path.exists(example_dir):
        os.makedirs(example_dir)
    link_dir = os.path.join(example_dir, os.path.basename(out_dir))
    if os.path.exists(link_dir):
        os.unlink(link_dir)
    os.symlink(out_dir, link_dir)

    if need_name_size:
        import cv2
        f = open(list_file, 'r')
        lines = f.readlines()
        f.close()

        f = open(name_size_file, 'w')

        for l in lines:
            l = l.strip().split()
            img_name = os.path.join(root_dir,'{}.jpg'.format(l[0]))
            #print img_name
            img = cv2.imread(img_name)
            height, width = img.shape[:2]
            f.write(l[0]+' '+str(height)+' '+str(width)+'\n')

        f.close()

