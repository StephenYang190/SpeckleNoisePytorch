import os
import cv2
import numpy as np


def load_image(path, image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = in_ / 255.0
    return in_


data_root = '/home2/tongda/data/speckleDATA/'
data_source = '/home2/tongda/data/speckleDATA/train.lst'
lst_source = '/home2/tongda/data/speckleDATA/train_new.lst'
num = 0
deleNum = 0
with open(data_source, 'r') as f:
    data_list = [x.strip() for x in f.readlines()]
    
    with open(lst_source, 'w') as out:
        for line in data_list:
            gt_name = line.split()[1]
            data_label = load_image(os.path.join(data_root, gt_name), 128)
            var = np.var(data_label)
            if not np.all(data_label < 0.05) and var > 0.01:
                num += 1
                out.write(line + '\n')
            else:
                deleNum += 1
                print("delete a line.")

print("Total number : %d" % (num + deleNum))
print("Use number : %d" % num)
print("Delete number : %d" % deleNum)