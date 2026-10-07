import torch
from torch import nn

class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.InstanceNorm2d(b),nn.LeakyReLU(.2),nn.Conv2d(b,b,3,padding=1,bias=False),nn.InstanceNorm2d(b),nn.LeakyReLU(.2))

class UNet(nn.Module):
    """Direct 2-channel zero-filled input to real image; no learned analysis prior."""
    def __init__(self):
        super().__init__()
        widths=[32,64,128,256]
        self.enc=nn.ModuleList();a=2
        for b in widths:self.enc.append(Block(a,b));a=b
        self.center=Block(256,512)
        self.up=nn.ModuleList();self.dec=nn.ModuleList();a=512
        for b in reversed(widths):
            self.up.append(nn.ConvTranspose2d(a,b,2,stride=2));self.dec.append(Block(2*b,b));a=b
        self.output=nn.Conv2d(32,1,1)
    def forward(self,x):
        skips=[]
        for block in self.enc:
            x=block(x);skips.append(x);x=nn.functional.avg_pool2d(x,2)
        x=self.center(x)
        for up,block,skip in zip(self.up,self.dec,reversed(skips)):
            x=block(torch.cat((up(x),skip),1))
        return self.output(x)
