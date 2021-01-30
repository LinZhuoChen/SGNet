import torch.nn as nn
import functools
import torch

from graphs.ops.modules.s_conv import SConv
from graphs.ops.libs import InPlaceABNSync

affine_par = True
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1,
                 deformable=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        if deformable == False:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        else:
            self.conv2 = SConv(planes, planes, kernel_size=3, stride=stride,
                                          padding=1, deformable_groups=1, no_bias=True)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.deformable = deformable

    def forward(self, input):
        x, S = input
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.deformable == False:
            out = self.conv2(out)
        else:
            out = self.conv2(out, S)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return [out, S]

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, deformable=True):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 64, layers[0], deformable=deformable, seg=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, deformable=deformable, seg=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=2, deformable=deformable, seg=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1),
                                       deformable=deformable, seg=True)

        self.dsn3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.dsn4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def make_up_conv_layer(self, block, in_channels, out_channels, batch_size):
        return block(in_channels, out_channels, batch_size)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, deformable=False, seg=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), deformable=deformable))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if seg == False:
                layers.append(block(self.inplanes, planes, dilation=dilation,
                                    multi_grid=generate_multi_grid(i, multi_grid), deformable=deformable))
            else:
                if i >= blocks-2:
                    layers.append(block(self.inplanes, planes, dilation=dilation,
                                        multi_grid=generate_multi_grid(i, multi_grid), deformable=deformable))
                else:
                    layers.append(block(self.inplanes, planes, dilation=dilation,
                                        multi_grid=generate_multi_grid(i, multi_grid), deformable=False))

        return nn.Sequential(*layers)
    def forward(self, x, depth):
        S = depth
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = self.maxpool(x)

        x = [x, S]

        x = self.layer3(self.layer2(self.layer1(x)))
        x3 = self.dsn3(x[0])

        x = self.layer4(x)
        x4 = self.dsn4(x[0])

        return [x4, x3]

    def load_pretrain(self, pretrain_model_path):
        """Load pretrained Network"""
        saved_state_dict = torch.load(pretrain_model_path)
        new_params = self.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

        self.load_state_dict(new_params)

def SGNet(num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model