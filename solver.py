import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from build_model import buildModel


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.build_model(config.op)


    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))


    # build the network
    def build_model(self, op):
        self.net = buildModel(op)

        if self.config.multi_gpu:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        else:
            self.net = self.net.to("cuda:0")

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'SNRNetwork_' + op)


    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        vali_num = len(self.test_loader)
        Maxvailoss = 100
        down = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.optimizer.zero_grad()
        writer = SummaryWriter("./tensorboard/" + self.config.op + '/')

        for epoch in range(self.config.epoch):
            TotalLoss = 0
            time_s = time.time()
            self.net.train()
            print("epoch: %2d/%2d || " % (epoch, self.config.epoch), end='')
            for i, data_batch in enumerate(self.train_loader):
                Noise_img, GT_img = data_batch['data_image'].cuda(), data_batch['data_label'].cuda()
                if (Noise_img.size(2) != GT_img.size(2)) or (Noise_img.size(3) != GT_img.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue

                output, low = self.net(Noise_img)

                loss_fun = nn.MSELoss()
                Loss = loss_fun(output, GT_img) + 4 * loss_fun(low, down(GT_img))
                TotalLoss += Loss
                Loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if (i + 1) % (iter_num / 20) == 0:
                    print('>', end='', flush=True)

            time_e = time.time()
            writer.add_scalar("Loss/train", TotalLoss / iter_num, epoch)
            self.net.eval()
            vailoss = 0
            loss_fun = nn.MSELoss()
            for i, data_batch in enumerate(self.test_loader):
                with torch.no_grad():
                    Noise_img, GT_img = data_batch['data_image'].cuda(), data_batch['data_label'].cuda()
                    if (Noise_img.size(2) != GT_img.size(2)) or (Noise_img.size(3) != GT_img.size(3)):
                        print('IMAGE ERROR, PASSING```')
                        continue

                    output, _ = self.net(Noise_img)
                    vailoss += loss_fun(output, GT_img)
            print(' || Loss : %10.4f || vailoss : %10.4f || Time : %f s' % (
            TotalLoss / iter_num, vailoss / vali_num, time_e - time_s))

            # write tensorboard
            writer.add_scalar("Train Loss", TotalLoss / iter_num, epoch)
            writer.add_scalar("Vaild Loss", vailoss / vali_num, epoch)
            if TotalLoss < Maxvailoss and epoch > 9:
                Maxvailoss = TotalLoss
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

        writer.flush()
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)