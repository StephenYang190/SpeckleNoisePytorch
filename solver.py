import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import time


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
            # print('Loading pre-trained model from %s...' % self.config.model)
            # self.net.load_state_dict(torch.load(self.config.model, map_location=config.device_id))
            # self.net.eval()
            pass
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
            from model.SNRo import SNRo
            self.net = SNRo(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                    alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
            self.testFolder = 'DenoiseResult-o/sythetics'
        elif self.config.op == 'om':
            from model.SNRom import SNRom
            self.net = SNRom(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                    alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
            self.testFolder = 'DenoiseResult-om/sythetics'
        elif self.config.op == 'c':
            from model.SNRc import SNRc
            self.net = SNRc(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3,
                    stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
            self.testFolder = 'DenoiseResult-c/sythetics'
        else:
            from model.SNRcm import SNRcm
            self.net = SNRcm(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3,
                    stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
            self.testFolder = 'DenoiseResult-cm/sythetics'

        if self.config.cuda:
            self.net.to(self.device)

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'SNRNetwork')

    def test(self):
        checkpoints = os.listdir(self.config.model)
        for checkpoint in checkpoints:
            if checkpoint == 'final.pth':
                continue
            iternum = checkpoint.split('_')[1].split('.')[0]
            outdir = os.path.join(self.config.test_folder, self.testFolder, iternum)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            print('Loading pre-trained model from %s...' % checkpoint)
            self.net.load_state_dict(torch.load(os.path.join(self.config.model, checkpoint), map_location=self.config.device_id))
            self.net.eval()

            time_s = time.time()
            img_num = len(self.test_loader)
            for i, data_batch in enumerate(self.test_loader):
                images, name = data_batch['image'], data_batch['name'][0]
                print("Proccess :" + name)

                with torch.no_grad():
                    images = images.to(self.device)

                    preds = self.net(images)
                    pred = np.squeeze(preds).cpu().data.numpy()
                    pred = pred * 255.0

                    filename = os.path.join(outdir, name[:-4] + '_' + self.config.op + '_denoise.png')
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

                output, x_l = self.net(Noise_img)

                loss_fun = nn.MSELoss()
                downs = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
                LossH = loss_fun(output, GT_img)
                LossL = loss_fun(x_l, downs(GT_img))
                Loss = LossH + LossL

                TotalLoss += LossH
                Loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()


                if (i + 1) % (iter_num / 20) == 0:
                    print('>', end='', flush=True)

            time_e = time.time()
            print(' || Loss : %10.4f || Time : %f s' % (TotalLoss / iter_num, time_e - time_s))
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)