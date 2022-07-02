import torch
from torch import nn
import torch.nn.functional as F
import time

from resnet import * 
from BasicModules import * 


class CCEModule(nn.Module):
    def __init__(self, backbone_res34):
        super(CCEModule, self).__init__()

        self.backbone = backbone_res34
        self.relu = nn.ReLU(inplace=True)

        cp = []
        cp.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), self.relu, aggregation_scale(64, 64)))
        cp.append(nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), self.relu, aggregation_scale(64, 64)))
        cp.append(nn.Sequential(nn.Conv2d(256, 96, 3, 1, 1), self.relu, aggregation_scale(96, 64)))
        cp.append(nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), self.relu, aggregation_scale(128, 64)))
        self.CP = nn.ModuleList(cp)
        
        # cross collaboration encoding 
        self.att_block2 = CCE(in_dim=64, sr_ratio=4)
        self.att_block3 = CCE(in_dim=128, sr_ratio=4)
        self.att_block4 = CCE(in_dim=256, sr_ratio=2)
        self.att_block5 = CCE(in_dim=512, sr_ratio=1)
        
    def load_pretrained_model(self, model_path):
        # resnet pretrained parameter
        pretrained_dict_res = torch.load(model_path)
        res_model_dict = self.backbone.state_dict()
        pretrained_dict_res = {k: v for k, v in pretrained_dict_res.items() if k in res_model_dict}
        res_model_dict.update(pretrained_dict_res)
        self.backbone.load_state_dict(res_model_dict)
        
        
    def forward(self, x):
        B = x.shape[0]
        feature_extract = []
        tmp_x = []
        
        ############################ stage 0 ###########################
        res1 = self.backbone.conv1(x)
        res1 = self.backbone.bn1(res1)
        res1 = self.backbone.relu(res1)           
        #tmp_x.append(res1)
        res1 = self.backbone.maxpool(res1)        # # 𝐻/4 x 𝑊/4 x 64
        
        ############################ stage 1 ###########################
        x1 = self.backbone.layer1(res1)        # 𝐻/4 x 𝑊/4 x 64
        x2 = self.att_block2(x1)
        tmp_x.append(x2)

        ########################### stage 2 ###########################
        x2 = self.backbone.layer2(x2)       # 𝐻/8 x 𝑊/8 x 128  
        x3 = self.att_block3(x2)
        tmp_x.append(x3)

        ############################ stage 3 ###########################
        x3 = self.backbone.layer3(x3)      # 𝐻/16 x 𝑊/16 x 256
        x4 = self.att_block4(x3)
        tmp_x.append(x4)

        ############################ stage 4 ###########################
        x4 = self.backbone.layer4(x4)      # 𝐻/32 x 𝑊/32 x512
        x5 = self.att_block5(x4)
        tmp_x.append(x5)
        

        for i in range(4):
            feature_extract.append(self.CP[i](tmp_x[i]))   

        return feature_extract                  

    
class CCFENet(nn.Module):
    def __init__(self, base_model_cfg, CCEModule, sp_layers):
        super(CCFENet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.JLModule = CCEModule

        self.cm = sp_layers
        self.score_JL = nn.Conv2d(64, 1, 1, 1)

        channel = 32
        self.rfb2_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.rfb3_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.rfb4_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.agg1 = aggregation_cross(channel)

        self.thm2_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.thm3_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.thm4_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.thm_agg1 = aggregation_cross(channel)


        self.inplanes = 32
        self.deconv1 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.inplanes =32
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32, 32)
        self.agant2 = self._make_agant_layer(32, 32)
        self.out0_conv = nn.Conv2d(32*3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(32*2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(32*1, 1, kernel_size=1, stride=1, bias=True)
        

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        x = self.JLModule(x)
        
        rgb2, rgb3, rgb4, rgb5, tma2, tma3, tma4, tma5 = self.cm(x)  
        s_coarse = self.score_JL(x[3])
        
        rgb3_1 = self.rfb2_1(rgb3)
        rgb4_1 = self.rfb3_1(rgb4)
        rgb5_1 = self.rfb4_1(rgb5)
        rgb_fea, rgb_map = self.agg1(rgb5_1, rgb4_1, rgb3_1)  
        y_rgb = rgb_map                                          
        
        tma3_1 = self.thm2_1(tma3)
        tma4_1 = self.thm3_1(tma4)
        tma5_1 = self.thm4_1(tma5)
        tma_fea, tma_map = self.thm_agg1(tma5_1, tma4_1, tma3_1) 
        y_thm = tma_map
        

        y = self.upsample2(rgb_fea + tma_fea)   
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)
        s_output = self.upsample(rgb_map) + self.upsample(tma_map) + y     
        
        return s_coarse, self.upsample(rgb_map), self.upsample(tma_map), y, s_output


    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),)
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),)
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        return nn.Sequential(*layers)


def build_model(base_model_cfg='resnet'):
    if base_model_cfg == 'resnet':
        backbone_res34 = ResNet(BasicBlock, [3, 4, 6, 3])        
        
        return CCFENet(base_model_cfg, CCEModule(backbone_res34), SPLayer())
