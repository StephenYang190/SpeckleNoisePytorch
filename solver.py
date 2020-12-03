import torch
import torch.nn as nn
import numpy as np
from model.SNRc import SNRc
from model.SNRo import SNRo
from model.SNRom import SNRom
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('log/run' + time.strftime("%d-%m"))


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.device = torch.device(config.device_id) if torch.cuda.is_available() else torch.device("cpu")
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()
        if config.mode == 'train':
            self.net.train()


    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        if self.config.op == 'o':
            self.net = SNRo(in_channels=3, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                    alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
        elif self.config.op == 'om':
            self.net = SNRom(in_channels=3, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                    alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
        elif self.config.op == 'c':
            self.net = SNRc(in_channels=3, hide_channels=64, out_channels=1, kernel_size=3,
                    stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
        else:
            self.net = SNRcm(in_channels=3, hide_channels=64, out_channels=1, kernel_size=3,
                    stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)

        if self.config.cuda:
            self.net.to(self.device)

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'SNRNetwork')

    def test(self):
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name = data_batch['image'], data_batch['name'][0]
            print("Proccess :" + name)
            
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)

                preds = self.net(images)
                pred = np.squeeze(preds).cpu().data.numpy()
                pred = pred * 255.0
                
                filename = os.path.join(self.config.test_folder, name[:-4] + '_denoise.png')
                cv2.imwrite(filename, pred)

        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        print(len(self.train_loader.dataset))
        self.optimizer.zero_grad()
        for epoch in range(self.config.epoch):
            TotalLoss = 0
            time_s = time.time()
            print("epoch: %2d/%2d || " % (epoch, self.config.epoch), end='')
            for i, data_batch in enumerate(self.train_loader):
                Noise_img, GT_img = data_batch['data_image'].to(self.device), data_batch['data_label'].to(self.device)
                if (Noise_img.size(2) != GT_img.size(2)) or (Noise_img.size(3) != GT_img.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    Noise_img, GT_img = Noise_img.to(device), GT_img.to(device)

                output = self.net(Noise_img)

                loss_fun = nn.MSELoss()
                Loss = loss_fun(output, GT_img)
                TotalLoss += Loss
                Loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()


                if (i + 1) % (iter_num / 20) == 0:
                    print('>', end='', flush=True)
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', Loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)

            time_e = time.time()
            print(' || Loss : %10.4f || Time : %f s' % (TotalLoss / iter_num, time_e - time_s))
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)