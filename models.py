import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim


class Attn(nn.Module):
    def __init__(self,channels,reduction_attn = 8, reduction_sc = 2): # Ratio should be 4: 1
        super(Attn,self).__init__()
        self.channels_attn = channels//reduction_attn # reduce cal c
        self.channels_sc = channels//reduction_sc # reduce cal 4c
        
        #ATTN consisted of query,key and value, structure design
        
        self.qconv = spectral_norm(nn.Conv2d(channels,self.channels_attn,kernel_size = 1,bias=False)) # 1 x 1 filter
        self.kconv = spectral_norm(nn.Conv2d(channels,self.channels_attn,kernel_size = 1,bias=False))
        self.vconv = spectral_norm(nn.Conv2d(channels,self.channels_sc,kernel_size = 1,bias=False)) # weight 
        self.attnconv = spectral_norm(nn.Conv2d(self.channels_sc,channels,kernel_size = 1,bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # initializing weights
        nn.init.orthogonal_(self.qconv.weight.data)
        nn.init.orthogonal_(self.kconv.weight.data)
        nn.init.orthogonal_(self.vconv.weight.data)
        nn.init.orthogonal_(self.attnconv.weight.data)
        
    def forward(self,x):
        
        batch,_,h,w = x.size() # original channel size is not important
        
        qx = self.qconv(x).view(batch,self.channels_attn,-1) # b x oc x h x w >>>>> b x c x hw 
        kx = F.max_pool2d(self.kconv(x),2).view(batch,self.channels_attn,-1) # b x c x h/2 x w/2 >>>>> b x c x hw/4
        
        attn = torch.bmm(kx.permute(0,2,1),qx) # b x hw/4 x c and b x c x hw >>>>>> b x hw/4 x hw
        attn = F.softmax(attn,dim=1)
        
        vx = F.max_pool2d(self.vconv(x),2).view(batch,self.channels_sc,-1) #b x c*4 x hw/4
        ##b x c*4 x hw/4 mul b x hw/4 x hw >>> b x 4c x hw
        attn = torch.bmm(vx,attn).view(batch,self.channels_sc,h,w) #b x c*4 x hw
        
        attn = self.attnconv(attn) # b,oc,h,w
        
        out = self.gamma * attn + x  # attn plus residual
        
        return out 
    
    
class Dblock(nn.Module):
    def __init__(self,in_channels,out_channels,downs = False,optimized = False):
        super(Dblock,self).__init__()
        self.downs = downs
        self.optimized = optimized
        self.learnable_sc = in_channels != out_channels or downs
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels,out_channels,kernel_size = 3,padding = 1,bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels,out_channels,kernel_size = 3,padding = 1,bias=False))
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels,out_channels,kernel_size = 1,bias=False))
            nn.init.orthogonal_(self.conv_sc.weight.data)
        self.relu = nn.ReLU()
        
        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        
    def _res(self,x):
        if not self.optimized:
            x = self.relu(x)
        x = self.conv2(self.relu(self.conv1(x)))
        if self.downs:
            x = F.avg_pool2d(x,2)
            
        return x
    
    def _shorcut(self,x):
        if self.learnable_sc:
            if self.optimized:
                x = self.conv_sc(F.avg_pool2d(x,2)) if self.downs else self.conv_sc(x) # reducing computation size first
            else:
                x = F.avg_pool2d(self.conv_sc(x),2) if self.downs else self.conv_sc(x)
        return x
    
    def forward(self,x):
        #print(self._shorcut(x).size(),self._res(x).size(),'DDDDDD')
        return self._shorcut(x) + self._res(x)
    

class Dis(nn.Module):
    def __init__(self,ch,n_grapheme = 168,n_vowel = 11,n_consonant = 7,use_attn = False):
        super(Dis,self).__init__()
        self.ch = ch
        
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.use_attn = use_attn
        
        self.b1 = Dblock(1,ch,downs=True,optimized=True)
        if use_attn:
            self.attn = Attn(ch)
        
        self.b2 = Dblock(ch,ch*2,downs=True,optimized=False)
        self.b3 = Dblock(ch*2,ch*4,downs=True,optimized=False)
        self.b4 = Dblock(ch*4,ch*8,downs=True,optimized=False)
        self.b5 = Dblock(ch*8+1,ch*8,downs=False,optimized=False)

        
        
        self.fc_1 = spectral_norm(nn.Linear(ch*8,n_grapheme,bias=False))
        self.fc_2 = spectral_norm(nn.Linear(ch*8, n_vowel,bias=False))
        self.fc_3 = spectral_norm(nn.Linear(ch*8, n_consonant, bias=False))
        

        nn.init.orthogonal_(self.fc_1.weight.data)
        nn.init.orthogonal_(self.fc_2.weight.data)
        nn.init.orthogonal_(self.fc_3.weight.data)
        
    def minibatch_std(self,x,group_size = 4,eps = 1e-8):
        shape = x.size() # 4 dimension tensor
        y = x.view(group_size,-1,shape[1],shape[2],shape[3]) # 4,b/4,c,h,w
        y -= torch.mean(y, dim=0, keepdim=True) # x - e(x) and keep same shape for subtract
        y = torch.mean(y.pow(2), dim=0) # b/4,c,h,w
        y = torch.sqrt(y + eps) # b/4,c,h,w
        y = torch.mean(y, dim=[1,2,3], keepdim=True) #b/4,1,1,1
        y = y.repeat(group_size, 1, shape[2], shape[3]) # b/4 * 4,1,h,w >>> b,1,h,w
        
        return torch.cat([x,y],dim = 1)
    
    def forward(self,x):
        h = self.b1(x)
        if self.use_attn:
            h = self.attn(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(self.minibatch_std(h))
        h = F.gelu(h)
        h = torch.sum(h, dim=(2,3)) # b,c 
        
        out_1 = self.fc_1(h) # b,c >>> b,1
        out_2 = self.fc_2(h) # b,c >>> b,1
        out_3 = self.fc_3(h) # b,c >>> b,1
        return out_1, out_2,out_3 # multitask !!