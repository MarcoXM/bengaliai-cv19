import pretrainedmodels
import torch.nn as nn 
import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Attention(nn.Module):
    def __init__(self, channels, reduction_attn=8, reduction_sc=2):
        super().__init__()
        self.channles_attn = channels // reduction_attn
        self.channels_sc = channels // reduction_sc
        self.maxpooling2d = nn.MaxPool2d(2,2)
        self.conv_query = spectral_norm(nn.Conv2d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_key = spectral_norm(nn.Conv2d(channels, self.channles_attn, kernel_size=1, bias=False))
        self.conv_value = spectral_norm(nn.Conv2d(channels, self.channels_sc, kernel_size=1, bias=False))
        self.conv_attn = spectral_norm(nn.Conv2d(self.channels_sc, channels, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        
        nn.init.orthogonal_(self.conv_query.weight.data)
        nn.init.orthogonal_(self.conv_key.weight.data)
        nn.init.orthogonal_(self.conv_value.weight.data)
        nn.init.orthogonal_(self.conv_attn.weight.data)

    def forward(self, x):
        batch, _, h, w = x.size()
        
        proj_query = self.conv_query(x).view(batch, self.channles_attn, -1)
        proj_key = self.maxpooling2d(self.conv_key(x)).view(batch, self.channles_attn, -1)
        
        attn = torch.bmm(proj_key.permute(0,2,1), proj_query)
        attn = F.softmax(attn, dim=1)
        
        proj_value = self.maxpooling2d(self.conv_value(x)).view(batch, self.channels_sc, -1)
        attn = torch.bmm(proj_value, attn)
        attn = attn.view(batch, self.channels_sc, h, w)
        attn = self.conv_attn(attn)
        
        out = self.gamma * attn + x
        
        return out

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, optimized=False):
        super().__init__()
        self.downsample = downsample
        self.optimized = optimized
        self.learnable_sc = in_channels != out_channels or downsample
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.relu = nn.ReLU()
        
        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        if self.learnable_sc:
            nn.init.orthogonal_(self.conv_sc.weight.data)
    
    def _residual(self, x):
        if not self.optimized:
            x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        
        return x
    
    def _shortcut(self, x):
        if self.learnable_sc:
            if self.optimized:
                x = self.conv_sc(F.avg_pool2d(x, 2)) if self.downsample else self.conv_sc(x)
            else:
                x = F.avg_pool2d(self.conv_sc(x), 2) if self.downsample else self.conv_sc(x)
        
        return x
    
    def forward(self, x):
        return self._shortcut(x) + self._residual(x)


class ResNet101(nn.Module):
    def __init__(self,pretrain = True):
        super(ResNet101,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained = None)
            
        # 2048 X 7 X 7
        self.attn = Attention(2048)
        self.block2 = DBlock(2048, 1024, downsample=True)
        self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=False)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(1024, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        x = self.model.features(x)
        x = self.attn(x)
        x1 = self.block2(x)
        l0 = self.fc1(torch.sum(self.swish(x1), dim = (2,3)))
        x = self.block3(x1)
        x = self.block4(x)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2 

class ResNet34(nn.Module):
    def __init__(self,pretrain = True):
        super(ResNet34,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)
            
        #  ### 512 x 7 x 7
        self.attn = Attention(512)
        #self.block2 = DBlock(2048, 1024, downsample=True)
        #self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=True)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(512, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        b = x.size(0)
        x = self.model.features(x)
        x1 = self.attn(x)
#         x = self.block2(x)
#         x = self.block3(x)
        x = self.block4(x1)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l0 = self.fc1(torch.sum(self.swish(x1),dim=(2,3)))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2

class SResnet(nn.Module):
    def __init__(self,pretrain = True):
        super(SResnet,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained = None)

        # 2048 X7X7
        self.attn = Attention(2048)
        self.block2 = DBlock(2048, 1024, downsample=True)
        self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=False)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(1024, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        x = self.model.features(x)
        x = self.attn(x)
        x1 = self.block2(x)
        l0 = self.fc1(torch.sum(self.swish(x1), dim = (2,3)))
        x = self.block3(x1)
        x = self.block4(x)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2 


class Pnasnet(nn.Module):
    def __init__(self,pretrain = True):
        super(Pnasnet,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__['pnasnet5large'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['pnasnet5large'](pretrained = None)
            
        # 4320 X 11 X 11
        self.attn = Attention(4320)
        self.block2 = DBlock(4320, 1024, downsample=True)
        self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=True)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(1024, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        x = self.model.features(x)
        x = self.attn(x)
        x1 = self.block2(x)
        l0 = self.fc1(torch.sum(self.swish(x1), dim = (2,3)))
        x = self.block3(x1)
        x = self.block4(x)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2 


class Inresnet(nn.Module): # 改好打开
    def __init__(self,pretrain = True):
        super(Inresnet,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained = None)
            
        #  ### 1536 X 8 X 8
        self.attn = Attention(1536)
        self.block2 = DBlock(1536, 1024, downsample=True)
        self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=False)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(1024, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        x = self.model.features(x)
        x = self.attn(x)
        x1 = self.block2(x)
        l0 = self.fc1(torch.sum(self.swish(x1), dim = (2,3)))
        x = self.block3(x1)
        x = self.block4(x)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2

class PolyNet(nn.Module):
    def __init__(self,pretrain = True):
        super(PolyNet,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__['polynet'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['polynet'](pretrained = None)
            
        # 2048 X 9 X 9
        self.attn = Attention(2048)
        self.block2 = DBlock(2048, 1024, downsample=True)
        self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=False)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(1024, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        x = self.model.features(x)
        x = self.attn(x)
        x1 = self.block2(x)
        l0 = self.fc1(torch.sum(self.swish(x1), dim = (2,3)))
        x = self.block3(x1)
        x = self.block4(x)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2


class SeNet(nn.Module):
    def __init__(self,pretrain = True):
        super(SeNet,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__['senet154'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['senet154'](pretrained = None)
            
        # 2048 X 7 X 7
        self.attn = Attention(2048)
        self.block2 = DBlock(2048, 1024, downsample=True)
        self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=False)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(1024, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        x = self.model.features(x)
        x = self.attn(x)
        x1 = self.block2(x)
        l0 = self.fc1(torch.sum(self.swish(x1), dim = (2,3)))
        x = self.block3(x1)
        x = self.block4(x)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2
        
        
        
class IcNetv4(nn.Module):
    def __init__(self,pretrain = True):
        super(IcNetv4,self).__init__()
        if pretrain:
            self.model = pretrainedmodels.__dict__['inceptionv4'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['inceptionv4'](pretrained = None)
            
        #  ### 1536 X 8 X 8
        self.attn = Attention(1536)
        self.block2 = DBlock(1536, 1024, downsample=False)
        self.block3 = DBlock(1024, 512, downsample=True)
        self.block4 = DBlock(512, 256, downsample=False)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(512, 168, bias=False))
        self.fc2 = spectral_norm(nn.Linear(256, 11, bias=False))
        self.fc3 = spectral_norm(nn.Linear(256, 7, bias=False))

        nn.init.orthogonal_(self.fc1.weight.data)
        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.orthogonal_(self.fc3.weight.data)
        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self,x):
        x = self.model.features(x)
        x = self.attn(x)
        x = self.block2(x)
        x1 = self.block3(x)
        l0 = self.fc1(torch.sum(self.swish(x1), dim = (2,3)))
        x = self.block4(x1)
        x = self.swish(x)
        x = torch.sum(x, dim=(2,3))
        l1 = self.fc2(x)
        l2 = self.fc3(x)

        return l0,l1,l2 