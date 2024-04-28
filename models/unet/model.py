import torch
import torch.nn as nn

from .parts import *

class UNet(nn.Module) :
    def __init__(self, num_classes = 2, bilinear = False) :
        super().__init__()
        self.bilinear = bilinear
        self.num_classes = num_classes
        self.layer1 = DoubleConv(3, 64)
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        self.layer5 = Down(512, 1024)
        
        self.layer6 = Up(1024, 512, bilinear = self.bilinear)
        self.layer7 = Up(512, 256, bilinear = self.bilinear)
        self.layer8 = Up(256, 128, bilinear = self.bilinear)
        self.layer9 = Up(128, 64, bilinear = self.bilinear)
        
        self.layer10 = nn.Conv2d(64, self.num_classes, kernel_size = 1)
        
    def forward(self, x) :
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        x6 = self.layer6(x5, x4)
        x6 = self.layer7(x6, x3)
        x6 = self.layer8(x6, x2)
        x6 = self.layer9(x6, x1)
        
        return self.layer10(x6)
    

class UNet_small(nn.Module):
    def __init__(self, num_classes = 2, bilinear = False) :
        super().__init__()
        self.bilinear = bilinear
        self.num_classes = num_classes
        self.layer1 = DoubleConv(3, 64)
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        
        self.layer5 = Up(512, 256, bilinear = self.bilinear)
        self.layer6 = Up(256, 128, bilinear = self.bilinear)
        self.layer7 = Up(128, 64, bilinear = self.bilinear)
        
        self.layer8 = nn.Conv2d(64, self.num_classes, kernel_size = 1)
        
    def forward(self, x) :
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x5 = self.layer5(x4, x3)
        x5 = self.layer6(x5, x2)
        x5 = self.layer7(x5, x1)
        
        return self.layer8(x5)
    

class UNet_very_small(nn.Module):
    def __init__(self, num_classes = 2, bilinear = False) :
        super().__init__()
        self.bilinear = bilinear
        self.num_classes = num_classes
        self.layer1 = DoubleConv(3, 32)
        self.layer2 = Down(32, 64)
        self.layer3 = Down(64, 128)
        
        self.layer4 = Up(128, 64, bilinear = self.bilinear)
        self.layer5 = Up(64, 32, bilinear = self.bilinear)
        
        self.layer6 = nn.Conv2d(32, self.num_classes, kernel_size = 1)
        
    def forward(self, x) :
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        x4 = self.layer4(x3, x2)
        x4 = self.layer5(x4, x1)
        
        return self.layer6(x4)

class UNet_tiny(nn.Module):
    def __init__(self, num_classes = 2, bilinear = False) :
        super().__init__()
        self.bilinear = bilinear
        self.num_classes = num_classes
        self.layer1 = DoubleConv(3, 8)
        self.layer2 = Down(8, 16)
        self.layer3 = Down(16, 32)
        
        self.layer4 = Up(32, 16, bilinear = self.bilinear)
        self.layer5 = Up(16, 8, bilinear = self.bilinear)
        
        self.layer6 = nn.Conv2d(8, self.num_classes, kernel_size = 1)
        
    def forward(self, x) :
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        x4 = self.layer4(x3, x2)
        x4 = self.layer5(x4, x1)
        
        return self.layer6(x4)