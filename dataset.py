import os

import cv2
import torch
from torch.utils import data

import numpy as np
import random

random.seed(10)


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list, image_size, device):
        self.data_root = data_root
        self.data_source = data_list
        self.image_size = image_size
        self.device = torch.device(device)

        with open(self.data_source, 'r') as f:
            self.data_list = [x.strip() for x in f.readlines()]

        self.data_num = len(self.data_list)

    def __getitem__(self, item):
        # data loading
        im_name = self.data_list[item % self.data_num].split()[0]
        gt_name = self.data_list[item % self.data_num].split()[1]

        data_image = load_image(os.path.join(self.data_root, im_name), self.image_size)
        data_label = load_image(os.path.join(self.data_root, gt_name), self.image_size)

        data_image = data_image.transpose((2, 0, 1))
        data_label = data_label.transpose((2, 0, 1))

        data_image = torch.from_numpy(data_image)
        data_label = torch.from_numpy(data_label)

        sample = {'data_image': data_image, 'data_label': data_label}
        return sample

    def __len__(self):
        return self.data_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image = load_image_test(os.path.join(self.data_root, self.image_list[item].split()[0]))
        image = image.transpose((2, 0, 1))
        image = torch.Tensor(image)
        
        return {'image': image, 'name': self.image_list[item % self.image_num].split()[0].split('/')[-1],}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list, config.image_size, config.device_id)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    elif mode == 'vail':
        dataset = ImageDataTrain(config.vail_root, config.vail_list, config.image_size, config.device_id)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    return data_loader


def load_image(path, image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = in_ / 255.0
    in_ = in_[:, :, np.newaxis]
    return in_


def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    in_ = np.array(im, dtype=np.float32)
    if in_.shape[0] % 2 != 0:
        in_ = cv2.resize(in_, (in_.shape[1], in_.shape[0] + 1))
    if in_.shape[1] % 2 != 0:
        in_ = cv2.resize(in_, (in_.shape[1] + 1, in_.shape[0]))
    in_ = in_ / 255.0
    in_ = in_[:, :, np.newaxis]
    return in_


def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ = in_ / 255.0
    return in_