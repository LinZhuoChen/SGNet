#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn.modules.utils import _pair
import functools
from graphs.ops.functions import ModulatedDeformConvFunction
from graphs.ops.modules import DeformConv
from torch.nn import functional as F
from graphs.ops.libs import InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1, no_bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.no_bias = no_bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()
        if self.no_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class SConv_xyz(ModulatedDeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1, no_bias=False):
        super(SConv_xyz, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups, no_bias)

        self.offset_generator = nn.Conv2d(64,
                                          self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride, self.stride),
                                          padding=(self.padding, self.padding),
                                          bias=True)
        self.weight_generator2 = nn.Conv2d(64, self.kernel_size[0] * self.kernel_size[1], kernel_size=1, stride=(1, 1),
                                           padding=(0, 0), bias=True)

        self.weight_generator1 = DeformConv(64, 64, (kernel_size, kernel_size), stride=stride,
                                       padding=padding, num_deformable_groups=deformable_groups)

        self.spatial_projector = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
        )
        self.init_offset()
    def init_offset(self):
        self.offset_generator.weight.data.zero_()
        self.offset_generator.bias.data.zero_()

    def forward(self, x, S):

        S = F.interpolate(input=S, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        S_ = self.spatial_projector(S)
        offset = self.offset_generator(S_)
        S_ = self.weight_generator1(S_)
        feature = S_
        mask = self.weight_generator2(feature)
        mask = torch.sigmoid(mask)
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(x, offset, mask, self.weight, self.bias)

class SConv_feature(ModulatedDeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1, no_bias=False):
        super(SConv_feature, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups, no_bias)

        self.offset_generator = nn.Conv2d(64,
                                          self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride, self.stride),
                                          padding=(self.padding, self.padding),
                                          bias=True)
        self.weight_generator2 = nn.Conv2d(64, self.kernel_size[0] * self.kernel_size[1], kernel_size=1, stride=(1, 1),
                                           padding=(0, 0), bias=True)

        self.weight_generator1 = DeformConv(64, 64, (kernel_size, kernel_size), stride=stride,
                                       padding=padding, num_deformable_groups=deformable_groups)

        self.spatial_projector = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
        )
        self.init_offset()
    def init_offset(self):
        self.offset_generator.weight.data.zero_()
        self.offset_generator.bias.data.zero_()

    def forward(self, x, S):
        if len(list(S.size())) == 4:
            S = F.interpolate(input=S[:, 2, :, :].unsqueeze(1), size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        else:
            S = F.interpolate(input=S.unsqueeze(1), size=(x.size()[2], x.size()[3]), mode='bilinear',
                              align_corners=True)
        S_ = self.spatial_projector(x)
        offset = self.offset_generator(S_)
        S_ = self.weight_generator1(S_, offset)
        feature = S_
        mask = self.weight_generator2(feature)
        mask = torch.sigmoid(mask)
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(x, offset, mask, self.weight, self.bias)

class SConv(ModulatedDeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1, no_bias=False):
        super(SConv, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups, no_bias)

        self.offset_generator = nn.Conv2d(64,
                                          self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride, self.stride),
                                          padding=(self.padding, self.padding),
                                          bias=True)
        self.weight_generator2 = nn.Conv2d(64, self.kernel_size[0] * self.kernel_size[1], kernel_size=1, stride=(1, 1),
                                           padding=(0, 0), bias=True)

        self.weight_generator1 = DeformConv(64, 64, (kernel_size, kernel_size), stride=stride,
                                       padding=padding, num_deformable_groups=deformable_groups)

        self.spatial_projector = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
        )
        self.init_offset()
    def init_offset(self):
        self.offset_generator.weight.data.zero_()
        self.offset_generator.bias.data.zero_()

    def forward(self, x, S):
        if len(list(S.size())) == 4:
            S = F.interpolate(input=S[:, 2, :, :].unsqueeze(1), size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        else:
            S = F.interpolate(input=S.unsqueeze(1), size=(x.size()[2], x.size()[3]), mode='bilinear',
                              align_corners=True)
        S_ = self.spatial_projector(S)
        offset = self.offset_generator(S_)
        S_ = self.weight_generator1(S_, offset)
        feature = S_
        mask = self.weight_generator2(feature)
        mask = torch.sigmoid(mask)
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(x, offset, mask, self.weight, self.bias)


class ASPPModule_Adaptive(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule_Adaptive, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features))

        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            InPlaceABNSync(inner_features))

        self.conv6 = SConv(features, inner_features, kernel_size=3, padding=1, no_bias=True, stride=1)
        self.bn6 = InPlaceABNSync(inner_features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 6, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )

        self.spatial_projector = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
        )


    def forward(self, x, S):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        feat6 = self.bn6(self.conv6(x, S))
        out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), 1)

        bottle = self.bottleneck(out)

        return bottle


class SConv_depth(ModulatedDeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1, no_bias=False):
        super(SConv_depth, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups, no_bias)

        self.offset_generator = nn.Conv2d(64,
                                          self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride, self.stride),
                                          padding=(self.padding, self.padding),
                                          bias=True)
        self.weight_generator2 = nn.Conv2d(64, self.kernel_size[0] * self.kernel_size[1], kernel_size=1, stride=(1, 1),
                                           padding=(0, 0), bias=True)

        self.weight_generator1 = DeformConv(64, 64, (kernel_size, kernel_size), stride=stride,
                                       padding=padding, num_deformable_groups=deformable_groups)

        self.spatial_projector = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.init_offset()
    def init_offset(self):
        self.offset_generator.weight.data.zero_()
        self.offset_generator.bias.data.zero_()

    def forward(self, x, S):
        if len(list(S.size())) == 4:
            S = F.interpolate(input=S[:, 2, :, :].unsqueeze(1), size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        else:
            S = F.interpolate(input=S.unsqueeze(1), size=(x.size()[2], x.size()[3]), mode='bilinear',
                              align_corners=True)
        S_ = self.spatial_projector(S)

        S_channel_weight = torch.sigmoid(self.fc_layer(S_))

        offset = self.offset_generator(S_)
        S_ = self.weight_generator1(S_, offset)
        feature = S_
        mask = self.weight_generator2(feature)
        mask = torch.sigmoid(mask)
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(x, offset, mask, self.weight, self.bias) * S_channel_weight