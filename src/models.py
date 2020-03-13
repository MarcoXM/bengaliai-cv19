import pretrainedmodels
import torch.nn as nn 
import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm
from efficientnet_pytorch import EfficientNet
import math

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

class Effinet(nn.Module):
    def __init__(self,pretrain = True):
        super(Effinet,self).__init__()
        if pretrain:
            self.model = EfficientNet.from_pretrained('efficientnet-b7')
        
            
        #  ### 2560 X 7 X 7
        self.attn = Attention(2560)
        self.block2 = DBlock(2560, 1024, downsample=False)
        self.block3 = DBlock(1024, 512, downsample=False)
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
        x = self.model.extract_features(x)
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


class Gnet(nn.Module):
    def __init__(self,pretrain = True):
        super(Gnet,self).__init__()
        if pretrain:
            self.model = ghost_net()
        
            
        #  ### 160 X 7 X 7
        self.attn = Attention(160)
        self.block2 = DBlock(160, 960, downsample=True)
        self.block3 = DBlock(960, 512, downsample=False)
        self.block4 = DBlock(512, 256, downsample=False)
        
        self.swish = Swish()
        
        self.fc1 = spectral_norm(nn.Linear(960, 168, bias=False))
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

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = 1280
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, s 
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__=='__main__':
    model = ghost_net()
    model.eval()
    print(model)
    input = torch.randn(32,3,224,224)
    y = model.features(input)
    print(y.size())