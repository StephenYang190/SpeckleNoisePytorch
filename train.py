import argparse
import os
from dataset import get_loader
import time
from solver import Solver


def main(config):
    train_loader = get_loader(config)
    vail_loader = get_loader(config, mode='vail')
    config.save_folder = os.path.join(config.save_folder, config.op, time.strftime("%d"))

    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    train = Solver(train_loader, vail_loader, config)
    train.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001)  # Learning rate resnet:1e-4
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--multi_gpu', type=bool, default=True)

    # Training settings
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')  # pretrained model
    parser.add_argument('--save_folder', type=str, default='./checkpoints/')
    parser.add_argument('--iter_size', type=int, default=80)

    # Train data
    parser.add_argument('--train_root', type=str, default='/home2/tongda/data/speckleDATA/')
    parser.add_argument('--train_list', type=str, default='/home2/tongda/data/speckleDATA/train.lst')
    parser.add_argument('--vail_root', type=str, default='/home2/tongda/data/speckleDATA/')
    parser.add_argument('--vail_list', type=str, default='/home2/tongda/data/speckleDATA/vail.lst')
    parser.add_argument('--op', type=str, default='om', choices=['om', 'o', 'c', 'cm'])

    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    main(config)
