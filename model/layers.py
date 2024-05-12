import torch.nn as nn

class SE_Block(nn.Module):
    def __init__(self, c, r=16,mid=0):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if mid==0:
            mid = c//r
        self.excitation = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1,groups=1,fusion=['se'],height=1):
        super(BasicBlock, self).__init__()
        self.fusion = fusion
        self.groups=groups
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3,height), padding=(1,0), stride=stride, groups=groups,bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3,1), padding=(1,0), groups=groups,bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.relu = nn.LeakyReLU(inplace=True)
        if 'se' in fusion:
            self.se = SE_Block(out_planes)
            self.usese=True
        else:
            self.usese=False
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            ds_group = 1 if ('rc' in fusion) else groups
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=ds_group),#groups=groups,
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.usese:
            out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out