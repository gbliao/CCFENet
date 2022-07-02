import torch
from torch.nn import functional as F
from network import build_model
import numpy as np
import os
import cv2
import time
from datetime import datetime
from data_prefetcher import DataPrefetcher
from skimage import img_as_ubyte
from imageio import imsave
import os.path as osp
from tqdm import tqdm
from tensorboardX import SummaryWriter
import logging
torch.set_num_threads(4)



class Solver(object):
    def __init__(self, test_loader, config):
        self.test_loader = test_loader
        self.config = config
        self.build_model()
        self.net.eval()

        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model))


        
    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd
        
        self.base, self.head = [], []
        self.base_name, self.head_name = [], []
        
        for name, param in self.net.named_parameters():
            if 'backbone.conv1' in name or 'backbone.bn1' in name or 'backbone_t.conv1' in name or 'backbone_t.bn1' in name:
                self.base_name.append(name)      
                self.base.append(param)
            elif 'backbone' in name or 'backbone_t' in name:
                self.base_name.append(name)
                self.base.append(param)
            else:
                self.head_name.append(name)
                self.head.append(param)

        self.optimizer = torch.optim.Adam([{'params':self.base}, {'params':self.head}], lr=self.lr, weight_decay=self.wd)

        
    def test(self):
        mark_time = []
        img_num = len(self.test_loader)
        for images, depth, _ , (H, W), name in self.test_loader:
            print("Testing...")
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device).float()
                    depth = depth.to(device).float()
                
                sal_input = torch.cat((images, depth), dim=0)
                start = time.time()
                s_coarse, _, _, _, y = self.net(sal_input)
                end = time.time()
                mark_time.append(end - start)
                preds = F.interpolate(y, tuple((H, W)), mode='bilinear', align_corners=True)       
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)             
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder,name[0][:-4]+ '.png')                
                cv2.imwrite(filename, multi_fuse)   
                
        time_sum = 0
        for i in mark_time:
            time_sum += i
        print("FPS: %f" % (1.0 / (time_sum / len(mark_time))))
        print('Test Done!')

        
    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)