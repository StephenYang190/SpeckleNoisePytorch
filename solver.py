import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from build_model import buildModel
from tqdm import tqdm


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

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, weight_decay=self.config.wd)

        self.print_network(self.net, 'SNRNetwork_' + op)

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        vali_num = len(self.test_loader)
        maxvalidLoss = 100
        down = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        writer = SummaryWriter("./tensorboard/" + self.config.op + '/')

        for epoch in range(self.config.epoch):
            trainLoss = 0
            time_s = time.time()
            self.net.train()
            print("epoch: %2d/%2d: " % (epoch, self.config.epoch))

            # Train
            for i, data_batch in enumerate(tqdm(self.train_loader)):
                # Clean loss
                self.optimizer.zero_grad()
                # Get data
                Noise_img, GT_img = data_batch['data_image'].cuda(), data_batch['data_label'].cuda()
                if (Noise_img.size(2) != GT_img.size(2)) or (Noise_img.size(3) != GT_img.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue

                output, low = self.net(Noise_img)

                loss_fun = nn.MSELoss()
                Loss = loss_fun(output, GT_img) + 2 * loss_fun(low, down(GT_img))
                trainLoss += Loss.item()
                Loss.backward()

                self.optimizer.step()

            time_e = time.time()

            # Valid
            self.net.eval()
            validLoss = 0
            loss_fun = nn.MSELoss()
            for i, data_batch in enumerate(self.test_loader):
                # Do not store compute graph
                with torch.no_grad():
                    # Get data
                    Noise_img, GT_img = data_batch['data_image'].cuda(), data_batch['data_label'].cuda()
                    if (Noise_img.size(2) != GT_img.size(2)) or (Noise_img.size(3) != GT_img.size(3)):
                        print('IMAGE ERROR, PASSING```')
                        continue

                    output, _ = self.net(Noise_img)
                    validLoss += loss_fun(output, GT_img).item()
            print('Train Loss : %10.4f || Valid Loss : %10.4f || Time : %f s' % (
                trainLoss / iter_num, validLoss / vali_num, time_e - time_s))

            # write tensorboard
            writer.add_scalar("Train Loss", trainLoss / iter_num, epoch)
            writer.add_scalar("Vaild Loss", validLoss / vali_num, epoch)
            if validLoss < maxvalidLoss and epoch > 9:
                maxvalidLoss = validLoss
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

        writer.flush()
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
