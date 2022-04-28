import torch
import torch.nn as nn
from torch.nn.functional import conv_transpose2d
import torch.nn.functional as F

import torchvision
import torchvision.models as models
from torchvision.models import resnet18, resnet34, resnet101, resnet152
from torchvision import transforms
from torchvision.models.resnet import Bottleneck
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import pandas as pd
import numpy as np

from skimage import data, transform
from PIL import Image
import matplotlib.pyplot as plt
from os import walk


class ResnetVisualizer(nn.Module):

    def __init__(self, resnet,weight_list):
        super(ResnetVisualizer, self).__init__()
        self.model = resnet
        self.weight_list = weight_list

        for name, child in self.model.named_children():
            if 'layer' in name:
                setattr(self, name, LayerVisualizer(name, child, weight_list))
        
        # For Deconv
        self.k7x7 = torch.ones((1,1,7,7))
        self.k3x3 = torch.ones((1,1,3,3))

    def forward(self, x):
        input = x
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        act1 = x.mean(1, keepdim=True)
        x = self.model.maxpool(x)

        x, vis1 = self.layer1(x)
        x, vis2 = self.layer2(x)
        x, vis3 = self.layer3(x)
        target_feature_map, vis4 = self.layer4(x)
        ## average pooling is better than max pooling
        x = self.model.avgpool(target_feature_map)

        vis = list(reversed([act1] + vis1 + vis2 + vis3 + vis4))
        prod = vis[0]
        for i in range(1, len(vis)):
            act = vis[i]
            
            if prod.shape != act.shape:
                prod = conv_transpose2d(prod, self.k3x3, stride=2, padding=1, output_padding=1)
            prod *= act

        # Resize to input image
        prod = conv_transpose2d(
                prod, self.k7x7, stride=2, padding=3, output_padding=1)
        return x.numpy().flatten(), prod, target_feature_map #* input.mean(1, keepdim=True)

class LayerVisualizer(nn.Module):
    
    def __init__(self, name, layer, weight_list):
        super(LayerVisualizer, self).__init__()
        self.name = name
        self.layer = layer
        self.weight_list = weight_list
        for name, child in self.layer.named_children():
            if self.name == "layer4":
                setattr(self, name, BlockVisualizer(name,child,True,weight_list))
            else:
                setattr(self, name, BlockVisualizer(name,child,False,weight_list))

    def forward(self, x):

        vis=[]
        for name, child in self.layer.named_children():
            block = getattr(self, name)
            x, prod = block(x)
            vis += prod
        return x, vis

class BlockVisualizer(nn.Module):

    def __init__(self,name, block, lastLayer,weight_list):
        super(BlockVisualizer, self).__init__()
        self.name = name
        self.block = block
        self.lastLayer = lastLayer
        self.weight_list = weight_list
        self.k3x3 = torch.ones((1,1,3,3))
        self.k1x1 = torch.ones((1,1,1,1))


    def forward(self, x):
        vis = []
        residual = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        vis += [out.mean(1,keepdim=True)]

        out = self.block.conv2(out)
        out = self.block.bn2(out)        
        
        if self.block.downsample is not None:
            residual = self.block.downsample(x)

        out += residual
        out = self.block.relu(out)

        if (self.lastLayer) == True and (self.name == '1'):
            heatmap = (out.permute(0,2,3,1) * self.weight_list).permute(0,3,1,2)
            heatmap = torch.mean(heatmap, 1, keepdim=True)
#             heatmap_max = heatmap.max(axis = 0)[0]
#             heatmap /= heatmap_max
#             heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            vis += [heatmap]
        else:
            vis += [out.mean(1,keepdim=True)]
        return out, [vis[-1]] #vis#[out.mean(1,keepdim=True)]
    
class FilenameDataset(Dataset):

    def __init__(self, files, transform=None):
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            return self.transform(sample)
        transform_default = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return transform_default(sample)
    
