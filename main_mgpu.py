import argparse
import os
from dataset import get_loader
from solver import Solver
import time


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        vail_loader = get_loader(config, mode='vail')

        if not os.path.exists("%s/demo-%s" % (config.save_folder, time.strftime("%d"))):
            os.mkdir("%s/demo-%s" % (config.save_folder, time.strftime("%d")))
        config.save_folder = "%s/demo-%s" % (config.save_folder, time.strftime("%d"))
        train = Solver(train_loader, vail_loader, config)
        train.train()
    elif config.mode == 'test':
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_folder): os.makedirs(config.test_folder)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001)  # Learning rate resnet:1e-4
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=str, default='cuda:0')

    # Training settings
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')  # pretrained model
    parser.add_argument('--save_folder', type=str, default='checkpoints/')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--iter_size', type=int, default=10)

    # Train data
    parser.add_argument('--train_root', type=str, default='../DataRoot/')
    parser.add_argument('--train_list', type=str, default='../DataRoot/train.lst')
    parser.add_argument('--vail_root', type=str, default='../DataRoot/')
    parser.add_argument('--vail_list', type=str, default='../DataRoot/vail.lst')
    parser.add_argument('--op', type=str, default='om', choices=['om', 'o', 'c', 'cm'])

    # Testing settings
    parser.add_argument('--model', type=str, default='checkpoints/omcheck/bnLoss/epoch_26.pth')  # Snapshot
    parser.add_argument('--test_folder', type=str, default='../test/')  # Test results saving folder
    parser.add_argument('--test_root', type=str, default='../DataRoot/TEST/sythetics/a005')
    parser.add_argument('--test_list', type=str, default='../DataRoot/TEST/sythetics/test.lst')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    main(config)
