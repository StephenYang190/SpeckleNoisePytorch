import argparse
import os
from dataset import get_loader
from test_solver import Solver


def main(config):

    test_loader = get_loader(config, mode='test')
    test = Solver(test_loader, config)
    test.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--image_size', type=int, default=128)

    # Testing settings
    parser.add_argument('--model', type=str, default='checkpoints/fdd/05/')  # Snapshot
    parser.add_argument('--test_folder', type=str, default='./TestResult/fdd')  # Test results saving folder
    parser.add_argument('--test_root', type=str, default='/home2/tongda/data/speckleDATA/TEST/sythetics/')
    parser.add_argument('--test_list', type=str, default='/home2/tongda/data/speckleDATA/TEST/sythetics/a001.lst')
    parser.add_argument('--op', type=str, default='fdd', choices=['om', 'o', 'c', 'cm'])

    config = parser.parse_args()

    main(config)
