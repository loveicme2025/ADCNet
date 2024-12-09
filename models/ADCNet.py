import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, einsum
from einops.layers.torch import Rearrange
import sys
from . import MobileNetV2
    
def init_weights(m):
    if isinstance(m, nn.Conv2d):

        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)     
        
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
            
class MDFF(nn.Module):
    def __init__(self,in_channel = [16,24,32,96,320]):
        super(MDFF, self).__init__()
        #c1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channel[0], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channel[1], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation1 = nn.Sequential(
            nn.Conv2d(in_channel[3]*2, in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        #c2
        self.conv1_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel[0], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channel[1], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channel[2], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation2 = nn.Sequential(
            nn.Conv2d(in_channel[3]*3, in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        # c3
        self.conv2_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel[1], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channel[2], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(in_channel[3], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation3 = nn.Sequential(
            nn.Conv2d(in_channel[3]*3, in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        #c4
        self.conv3_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel[2], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv4_4 = nn.Sequential(
            nn.Conv2d(in_channel[3], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv5_4 = nn.Sequential(
            nn.Conv2d(in_channel[4], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation4 = nn.Sequential(
            nn.Conv2d(in_channel[3]*3, in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        
        #c5
        self.conv4_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channel[3], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_channel[4], in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation5 = nn.Sequential(
            nn.Conv2d(in_channel[3]*2, in_channel[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel[3]),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_channel[0], in_channel[3], kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel[1], in_channel[3], kernel_size=1)
        self.conv3 = nn.Conv2d(in_channel[2], in_channel[3], kernel_size=1)
        self.conv4 = nn.Conv2d(in_channel[3], in_channel[3], kernel_size=1)
        self.conv5 = nn.Conv2d(in_channel[4], in_channel[3], kernel_size=1)
    def forward(self,c1,c2,c3,c4,c5):
        # 1 [1,2]
        c1_1 = self.conv1_1(c1)
        c2_1 = self.conv2_1(c2)
        c2_1 = F.interpolate(c2_1, scale_factor=(2, 2), mode='bilinear')

        s1 = self.conv_aggregation1(torch.cat([c1_1, c2_1], dim=1))+self.conv1(c1)

        # 2 [1,2,3]
        c1_2 = self.conv1_2(c1)
        c2_2 = self.conv2_2(c2)
        c3_2 = self.conv3_2(c3)
        c3_2 = F.interpolate(c3_2, scale_factor=(2, 2), mode='bilinear')
        s2 = self.conv_aggregation2(torch.cat([c1_2, c2_2,c3_2], dim=1))+self.conv2(c2)
        # 3 [2,3,4]
        c2_3 = self.conv2_3(c2)
        c3_3 = self.conv3_3(c3)
        c4_3 = self.conv4_3(c4)
        c4_3 = F.interpolate(c4_3, scale_factor=(2, 2), mode='bilinear')
        s3 = self.conv_aggregation3(torch.cat([c2_3,c3_3,c4_3], dim=1))+self.conv3(c3)
        # 4 [3,4,5]
        c3_4 = self.conv3_4(c3)
        c4_4 = self.conv4_4(c4)
        c5_4 = self.conv5_4(c5)
        c5_4 = F.interpolate(c5_4, scale_factor=(2, 2), mode='bilinear')
        s4 = self.conv_aggregation4(torch.cat([c3_4,c4_4,c5_4], dim=1))+self.conv4(c4)
        #5 [4,5]
        c4_5 = self.conv4_5(c4)
        c5_5 = self.conv5_5(c5)
        s5 = self.conv_aggregation5(torch.cat([c4_5,c5_5], dim=1))+self.conv5(c5)
        
        return s1,s2, s3, s4, s5
        
class PG(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2
        self.conv5_4 = nn.Conv2d(in_channels[2], in_channels[1], 1)
        self.conv4_0 = BasicConv2d(in_channels[1], in_channels[1], 3)
        self.conv4_3 = nn.Conv2d(in_channels[1], in_channels[0], 1)
        self.conv3_0 = BasicConv2d(in_channels[0], in_channels[0], 3)
        self.conv_out = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.aff4 = AFS(channels=96)
        self.aff3 = AFS(channels=32)

    def forward(self, x3, x4, x5):
        x5_up = self.conv5_4(F.interpolate(x5, size=x4.shape[2:], mode='bilinear'))
        x4_refine = self.conv4_0(self.aff4(x4, x5_up))
        x4_up = self.conv4_3(F.interpolate(x4_refine, size=x3.shape[2:], mode='bilinear'))
        x3_refine = self.conv3_0(self.aff3(x3, x4_up))
        out = self.conv_out(x3_refine)
        return out

#head_list = ['fcn', 'parallel']
#head_list = ['fcn', 'parallel']

#norm_dict = {'BATCH': nn.BatchNorm2d, 'INSTANCE': nn.InstanceNorm2d, 'GROUP': nn.GroupNorm}

#class Flatten(nn.Module):
#    def forward(self, x):
#        return x.view(x.size(0), -1)

class BasicConv_do_eval(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do_eval, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
        
class ResBlock_do_fft_bench_eval(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_eval, self).__init__()
        self.main1 = nn.Sequential(
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft1 = nn.Sequential(
            BasicConv_do_eval(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do_eval(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm  
        
    def forward(self, x1):
        _, _, H, W = x1.shape
        y1 = torch.fft.rfft2(x1, norm=self.norm)
        y1_imag = y1.imag
        y1_real = y1.real
        y1_f = torch.cat([y1_real, y1_imag], dim=1)
        y1 = self.main_fft1(y1_f)
        y1_real, y1_imag = torch.chunk(y1, 2, dim=1)
        y1 = torch.complex(y1_real, y1_imag)
        y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)
        out = self.main1(x1) + x1 + y1
        
        return out
        
class AFS(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFS, self).__init__()
        inter_channels = int(channels // r)
        self.fft_encode = ResBlock_do_fft_bench_eval(channels)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x, residual):
        xa = x + residual
        xf = self.fft_encode(xa)
        xf = self.conv(xf)
        wei = self.sigmoid(xf)
        xo = x * wei + residual * (1 - wei)
        xo = self.conv_cat(torch.cat([xo, x, residual], dim=1))
        return xo 
        
class decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(decode, self).__init__()
        self.conv = BasicConv2d(in_channel, out_channel, 3)
        self.conv1 = BasicConv2d(out_channel*2, out_channel, 3)
        self.spc = AFHA(out_channel)
        
    def forward(self, left, down):
        down_mask = self.conv(down)
        down_ = F.interpolate(down_mask, scale_factor=(2,2), mode='bilinear')
        out = self.conv1(torch.cat([down_ ,left],dim=1))
        out = self.spc(out)
        
        return out
            
class AFSA(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()

        self.channel = k

        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.vert_pool = nn.AdaptiveAvgPool2d((1, None))
        self.hori_pool = nn.AdaptiveAvgPool2d((None, 1))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.gamma = nn.Parameter(torch.zeros(k,1,1))
        self.beta = nn.Parameter(torch.ones(k,1,1))
        self.conv = nn.Conv2d(k,k,1)
    def forward(self, x):
    
        pool = self.pool(x)
        pool = self.conv(pool)
        pool = torch.sigmoid(pool)
        
        hori_l = self.hori_pool(x) # 1,3,10,1
        hori_h = x - hori_l
        
        hori_out = self.hori_low * hori_l + self.hori_high * hori_h
        
        vert_l = self.vert_pool(x) # 1,3,1,10
        vert_h = x - vert_l

        vert_out = self.vert_low * vert_l + self.vert_high * vert_h
        out = pool*hori_out + (1-pool)*vert_out
        out = out + x
        
        return out
        
class MLCA1(nn.Module):
    def __init__(self, cur_channel ) -> None:
        super().__init__()
        
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=5, dilation=5) 
        
        self.cur_all = BasicConv2d(3 * cur_channel, cur_channel, 3, padding=1)

    def forward(self, x):
        x_cur_1 = self.cur_b1(x)
        x_cur_2 = self.cur_b2(x)
        x_cur_3 = self.cur_b3(x)
      
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3), 1))

        return x_cur_all
        
class MLCA2(nn.Module):
    def __init__(self, cur_channel ) -> None:
        super().__init__()
        
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=6, dilation=6)   
        
        self.cur_all = BasicConv2d(3 * cur_channel, cur_channel, 3, padding=1)

    def forward(self, x):
        x_cur_1 = self.cur_b1(x)
        x_cur_2 = self.cur_b2(x)
        x_cur_3 = self.cur_b3(x)

        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3), 1))

        return x_cur_all
        
class AFHA(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()

        self.global_att = AFSA(k)
        self.local_att_3 = MLCA1(k)
        self.local_att_5 = MLCA2(k)

        self.conv = nn.Conv2d(k, k, 1)

    def forward(self, x):
        
        global_out = self.global_att(x)
        local_3_out = self.local_att_3(x)
        local_5_out = self.local_att_5(x)

        out = global_out + local_3_out + local_5_out

        return self.conv(out)+ x
        
        
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation = dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class ADCNet(nn.Module):
    def __init__(self, num_classes=2, normal_init=True, pretrained=True):
        super(ADCNet, self).__init__()
        
        self.backbone = MobileNetV2.MobileNetV2(pretrained=True)  # [16, 24, 32, 96,320]
        
        encoder_channels = [16, 24, 32, 96, 320]
        self.one_stage = PG(in_channels=encoder_channels[2:], out_channels=1)
        self.conv_down = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
                                   
        self.MDFF = MDFF()
          
        self.conv1 = nn.Sequential(nn.Conv2d(192, 96, 3, stride=1,padding=1, bias=False),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(192, 96, 3, stride=1,padding=1, bias=False),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 96, 3,stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(192, 96, 3, stride=1,padding=1, bias=False),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True)) 
        self.conv5 = nn.Sequential(nn.Conv2d(192, 96, 3, stride=1,padding=1, bias=False),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True)) 
        self.fam1 = decode(96,96)
        self.fam2 = decode(96,96)
        self.fam3 = decode(96,96)
        self.fam4 = decode(96,96)
        self.fam5 = decode(96,96)
        self.fam6 = decode(96,96)
        self.fam7 = decode(96,96)
        self.fam8 = decode(96,96)
        self.fam9 = decode(96,96)
        self.fam10 = decode(96,96)

        self.final = nn.Sequential(
            Conv(96, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.fina2 = nn.Sequential(
            Conv(96, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.fina3 = nn.Sequential(
            Conv(96, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        self.fina4 = nn.Sequential(
            Conv(96, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
            )
        if normal_init:
            self.init_weights()
    
    def attentionPG(self,attention1,x1_5,x1_4,x1_3,x1_2,x1_1):
        act_attention1 = torch.sigmoid(attention1)
        act_attention11 = self.conv_down(act_attention1)
        attention5_1 = F.interpolate(act_attention11, size=x1_5.shape[2:], mode='bilinear', align_corners=False)
        attention4_1 = F.interpolate(act_attention11, size=x1_4.shape[2:], mode='bilinear', align_corners=False)
        attention3_1 = F.interpolate(act_attention1, size=x1_3.shape[2:], mode='bilinear', align_corners=False)
        attention2_1 = F.interpolate(act_attention1, size=x1_2.shape[2:], mode='bilinear', align_corners=False)
        attention1_1 = F.interpolate(act_attention1, size=x1_1.shape[2:], mode='bilinear', align_corners=False)

        x1_1, x1_2, x1_3, x1_4, x1_5 = self.MDFF(x1_1,x1_2,x1_3,x1_4,x1_5)

        x1_5 = x1_5*attention5_1 + x1_5
        x1_4 = x1_4*attention4_1 + x1_4
        x1_3 = x1_3*attention3_1 + x1_3
        x1_2 = x1_2*attention2_1 + x1_2
        x1_1 = x1_1*attention1_1 + x1_1
        
        return x1_1, x1_2, x1_3, x1_4, x1_5
        
    def forward(self, imgs1, imgs2, labels=None):        
        x1_1,x1_2,x1_3,x1_4,x1_5 = self.backbone(imgs1)
        x2_1,x2_2,x2_3,x2_4,x2_5 = self.backbone(imgs2) 
        
        attention1 = self.one_stage(x1_3, x1_4, x1_5)
        attention2 = self.one_stage(x2_3, x2_4, x2_5)
        
        x1_11, x1_21, x1_31, x1_41, x1_51 = self.attentionPG(attention1, x1_5, x1_4, x1_3, x1_2, x1_1)
        x2_12, x2_22, x2_32, x2_42, x2_52 = self.attentionPG(attention2, x2_5, x2_4, x2_3, x2_2, x2_1)

        x5 = self.conv5(torch.cat([x1_51,x2_52],dim=1))
        x4 = self.conv4(torch.cat([x1_41,x2_42],dim=1))
        x3 = self.conv3(torch.cat([x1_31,x2_32],dim=1))
        x2 = self.conv2(torch.cat([x1_21,x2_22],dim=1))
        x1 = self.conv1(torch.cat([x1_11,x2_12],dim=1))
        
        out1 = self.fam1(x4, x5)
        out2 = self.fam2(x3, out1)
        out3 = self.fam3(out2, out1)
        out4 = self.fam4(x2, out2)
        out5 = self.fam5(out4, out3)      
        out6 = self.fam6(out5, out3) 
        out7 = self.fam7(x1, out4) 
        out8 = self.fam8(out7, out5) 
        out9 = self.fam9(out8, out6) 
        out10 = self.fam10(out9, out6) 

        out_10 = self.final(F.interpolate(out10, scale_factor=(2, 2), mode='bilinear'))
        out_9 = self.final(F.interpolate(out9, scale_factor=(2, 2), mode='bilinear'))
        out_8 = self.final(F.interpolate(out8, scale_factor=(2, 2), mode='bilinear'))
        out_7 = self.final(F.interpolate(out7, scale_factor=(2, 2), mode='bilinear'))
        out_6 = self.fina2(F.interpolate(out6, scale_factor=(4, 4), mode='bilinear'))
        out_5 = self.fina2(F.interpolate(out5, scale_factor=(4, 4), mode='bilinear'))
        out_4 = self.fina2(F.interpolate(out4, scale_factor=(4, 4), mode='bilinear'))
        out_3 = self.fina3(F.interpolate(out3, scale_factor=(8, 8), mode='bilinear'))
        out_2 = self.fina4(F.interpolate(out2, scale_factor=(8, 8), mode='bilinear'))
        out_1 = self.fina4(F.interpolate(out1, scale_factor=(16, 16), mode='bilinear'))
            
        return out_10, out_6, out_3, out_1
    def init_weights(self): 

        self.MDFF.apply(init_weights)
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.conv3.apply(init_weights)
        self.conv4.apply(init_weights)
        
        self.fam2.apply(init_weights)
        self.fam3.apply(init_weights)
        self.fam4.apply(init_weights)

        self.final.apply(init_weights)
        self.fina2.apply(init_weights)
        self.fina3.apply(init_weights)
        self.fina4.apply(init_weights)
        