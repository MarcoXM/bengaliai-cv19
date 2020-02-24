import pretrainedmodels
import torch.nn as nn 
import torch.functional as F


class ResNet152(nn.Module):
    def __init__(self,pretrain = True):
        super(ResNet152,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__["resnet152"](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__["resnet152"](pretrained = None)

        self.l0 = nn.Linear(2048,168)
        self.l1 = nn.Linear(2048,11)
        self.l2 = nn.Linear(2048,7)
        self.a = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self,x):
        b = x.size(0)
        x = self.model.features(x)
        x = self.a(x).view(b,-1)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1,l2 

class ResNet34(nn.Module):
    def __init__(self,pretrain = True):
        super(ResNet34,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained = None)

        self.l0 = nn.Linear(512,168)
        self.l1 = nn.Linear(512,11)
        self.l2 = nn.Linear(512,7)
        self.a = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self,x):
        b = x.size(0)
        x = self.model.features(x)
        x = self.a(x).view(b,-1)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1,l2 

class SResnet(nn.Module):
    def __init__(self,pretrain = True):
        super(SResnet,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained = None)

        self.l0 = nn.Linear(2048,168)
        self.l1 = nn.Linear(2048,11)
        self.l2 = nn.Linear(2048,7)
        self.a = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self,x):
        b = x.size(0)
        x = self.model.features(x)
        x = self.a(x).view(b,-1)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1,l2 


class Pnasnet(nn.Module):
    def __init__(self,pretrain = True):
        super(Pnasnet,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__["pnasnet5large"](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__["pnasnet5large"](pretrained = None)

        self.l0 = nn.Linear(4320,168)
        self.l1 = nn.Linear(4320,11)
        self.l2 = nn.Linear(4320,7)
        self.a = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self,x):
        b = x.size(0)
        x = self.model.features(x)
        x = self.a(x).view(b,-1)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1,l2 

