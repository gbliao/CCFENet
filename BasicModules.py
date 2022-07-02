import torch
from torch import nn
import torch.nn.functional as F
import time

    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SPLayer(nn.Module):
    def __init__(self):
        super(SPLayer, self).__init__()

    def forward(self, list_x):
        rgb0 = list_x[0][0].unsqueeze(dim=0)
        tma0 = list_x[0][1].unsqueeze(dim=0)
        rgb1 = list_x[1][0].unsqueeze(dim=0)
        tma1 = list_x[1][1].unsqueeze(dim=0)
        rgb2 = list_x[2][0].unsqueeze(dim=0)
        tma2 = list_x[2][1].unsqueeze(dim=0)
        rgb3 = list_x[3][0].unsqueeze(dim=0)
        tma3 = list_x[3][1].unsqueeze(dim=0)

        return rgb0, rgb1, rgb2, rgb3, tma0, tma1, tma2, tma3   

    
class CA(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

    
class CCE(nn.Module):
    def __init__(self, in_dim=2048, sr_ratio=1):
        super(CCE, self).__init__()
        input_dim = in_dim
        self.chanel_in = input_dim  
        
        self.query_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        
        self.query_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        
        self.sr_ratio = sr_ratio
        dim = in_dim
        
        if sr_ratio > 1:
            self.sr_k = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_v = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_k = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_v = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            
            self.sr_kk = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_vv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_kk = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_vv = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)            
            
        self.gamma_rd = nn.Parameter(torch.zeros(1))
        self.gamma_dr = nn.Parameter(torch.zeros(1))
        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(dim*2, dim//2, kernel_size = 1)  
        self.fc2 = nn.Conv2d(dim//2, dim*2, kernel_size = 1)
        self.merge_conv1x1 = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, 1), self.relu)

        
    def forward(self, x):   
        xr, xd = x[0].unsqueeze(dim=0), x[1].unsqueeze(dim=0)
        m_batchsize, C, width, height = xr.size()

        query_r = self.query_convrd(xr).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        if self.sr_ratio > 1:   
            key_d = self.norm_k(self.sr_k(xd))       
            value_d = self.norm_v(self.sr_v(xd))
            key_d = self.key_convrd(key_d).view(m_batchsize, -1, width//self.sr_ratio * height//self.sr_ratio)
            value_d = self.value_convrd(value_d).view(m_batchsize, -1, width//self.sr_ratio * height//self.sr_ratio)
        else:
            key_d = self.key_convrd(xd).view(m_batchsize, -1, width * height)
            value_d = self.value_convrd(xd).view(m_batchsize, -1, width * height) 
        attention_rd = self.softmax(torch.bmm(query_r, key_d))
        out_rd = torch.bmm(value_d, attention_rd.permute(0, 2, 1))
        out_rd = out_rd.view(m_batchsize, C, width, height)
        

        query_d = self.query_convdr(xd).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        if self.sr_ratio > 1:    
            key_r = self.norm_kk(self.sr_kk(xr))       
            value_r = self.norm_vv(self.sr_vv(xr))
            key_r = self.key_convdr(key_r).view(m_batchsize, -1, width//self.sr_ratio * height//self.sr_ratio)
            value_r = self.value_convdr(value_r).view(m_batchsize, -1, width//self.sr_ratio * height//self.sr_ratio)
        else:
            key_r = self.key_convdr(xr).view(m_batchsize, -1, width * height)
            value_r = self.value_convdr(xr).view(m_batchsize, -1, width * height) 
        attention_dr = self.softmax(torch.bmm(query_d, key_r))
        out_dr = torch.bmm(value_r, attention_dr.permute(0, 2, 1))
        out_dr = out_dr.view(m_batchsize, C, width, height)        
        
        out_rd = self.gamma_rd * out_rd + xr
        out_dr = self.gamma_dr * out_dr + xd
        out_rd = self.relu(out_rd)
        out_dr = self.relu(out_dr)
        

        rgb_gap = nn.AvgPool2d(out_rd.shape[2:])(out_rd).view(len(out_rd), C, 1, 1)
        hha_gap = nn.AvgPool2d(out_dr.shape[2:])(out_dr).view(len(out_dr), C, 1, 1)
        stack_gap = torch.cat([rgb_gap, hha_gap], dim=1)   
        stack_gap = self.fc1(stack_gap)
        stack_gap = self.relu(stack_gap)
        stack_gap = self.fc2(stack_gap)        
        rgb_ = stack_gap[:, 0:C, :, :] * out_rd
        hha_ = stack_gap[:, C:2*C, :, :] * out_dr
        merge_feature = torch.cat([rgb_, hha_], dim=1)
        merge_feature = self.merge_conv1x1(merge_feature)

        rgb_out = (xr + merge_feature) / 2
        hha_out = (xd + merge_feature) / 2
        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)
        
        out_x = torch.cat([rgb_out, hha_out], dim=0)

        return out_x


   
class aggregation_scale(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=[1,2,3], residual=False):
        super(aggregation_scale, self).__init__()

        if in_dim == out_dim:
            residual=True
        self.use_res_connect = residual
        mid_dim = out_dim*2

        self.conv1 = BasicConv2d(in_dim, mid_dim, kernel_size=1)
        self.hidden_conv1 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=mid_dim, dilation=1)
        self.hidden_conv2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=2, groups=mid_dim, dilation=2)
        self.hidden_conv3 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=3, groups=mid_dim, dilation=3)
        
        self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(mid_dim), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(nn.Conv2d(mid_dim, out_dim, 1, 1, 0, bias=False))
        
    def forward(self, input_):
        x = self.conv1(input_)
        x1 = self.hidden_conv1(x) 
        x2 = self.hidden_conv2(x) 
        x3 = self.hidden_conv3(x)
        intra = self.hidden_bnact(x1+x2+x3)
        output = self.out_conv(intra)
    
        if self.use_res_connect:
            output = input_ + output

        return output 

    
class aggregation_cross(nn.Module):
    def __init__(self, channel):
        super(aggregation_cross, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)
        
    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x_k = self.conv4(x3_2)     
        x = self.conv5(x_k)
        return x_k, x

    
class TransBasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out