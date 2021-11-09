import torch
import numpy as np
import os
import cv2
import time
from build_model import buildModel


class Solver(object):
    def __init__(self, test_loader, config):
        self.test_loader = test_loader
        self.config = config
        self.build_model(config.op)


    # build the network
    def build_model(self, op):
        self.net = buildModel(op)
        self.net = self.net.to("cuda:0")


    def test(self):
        files = os.listdir(self.config.model)
        for file in files:
            if not file.endswith(".pth"):
                continue
            outdir = os.path.join(self.config.test_folder, file.split('.')[0])
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            epoch_name = os.path.join(self.config.model, file)

            self.load_dict(epoch_name)
            self.net.eval()
            self.test_no_metric(outdir)


    def test_no_metric(self, outdir):
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name = data_batch['image'], data_batch['name'][0]
            print("Proccess :" + name)

            with torch.no_grad():
                images = images.to('cuda:0')

                preds, _ = self.net(images)
                pred = np.squeeze(preds).cpu().data.numpy()
                pred = pred * 255.0

                # filename = os.path.join(outdir, name[:-4] + '_' + self.config.op + '_denoise.png')
                filename = os.path.join(outdir, name)
                cv2.imwrite(filename, pred)

        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')


    def load_dict(self, epoch_name):
        print('Loading pre-trained model from %s...' % epoch_name)
        # original saved file with DataParallel
        state_dict = torch.load(epoch_name, map_location='cuda:0')
        # create new OrderedDict that does not contain `module.`
        frsKey, _ = list(state_dict.items())[0]
        if 'module' in frsKey:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            self.net.load_state_dict(new_state_dict)
        else:
            self.net.load_state_dict(state_dict)