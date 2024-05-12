import torch
import torch.nn.functional as F

from .layers import *

class GRNet(nn.Module):
    def __init__(
        self,data_shape=(1,16384,2),expand=[2*4,2*4,4*4,8*4,16*4],num_classes=100,fusion=[],loss_type="cel"):
        super(GRNet, self).__init__()
        self.in_planes = data_shape[0]
        self.usefpn = False
        self.groups = data_shape[0]
        self.fusion=fusion
        self.usese=False
        self.quad=False
        self.num_classes = num_classes
        num_blocks = 2
        if 'se' in self.fusion:
            self.usese = True
        if loss_type not in ["cel"]:
            raise ValueError("__init__() got unknown loss type")
        self.loss_type=loss_type
        self.layers=nn.ModuleList()
        self.layers.append(nn.Sequential(
                nn.Conv2d(self.in_planes, self.in_planes*expand[0], kernel_size=(7,2), stride=1, padding=0, bias=False,groups=self.groups),
                nn.BatchNorm2d(self.in_planes*expand[0]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3,1), stride=2, padding=0)
            ))
        now_channels = self.in_planes*expand[0]
        strides_layers = [1,1,2,2]
        for i, mult in enumerate(expand[1:]):
            out_channels = self.in_planes * mult
            strides_blocks = [strides_layers[i]] +[1]*(num_blocks-1)
            for j in range(num_blocks):
                self.layers.append(BasicBlock(
                    now_channels,
                    out_channels,
                    groups=self.groups,
                    stride=strides_blocks[j],
                    fusion=fusion,
                ))
                now_channels = out_channels
        
        self.layers.append(nn.AdaptiveAvgPool2d((1,1)))
        self.fc = nn.Linear(now_channels, self.num_classes)
        self.loss_type = loss_type
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
    
    def get_loss(self, pred,y=None,reduction='mean'):
        if self.loss_type == "cel":
            loss = F.cross_entropy(pred, y, reduction=reduction)
        return loss

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    import thop
    m = GRNet(data_shape=(128,64,2),num_classes=10).cuda()
    print(m)
    x=torch.randn((1,128,64,2)).cuda()
    y=m(x).cuda()
    print(x.size())
    print(y.size())
    flops, params = thop.profile(m,inputs= (x,))
    flops, params =thop.clever_format([flops, params],"%.3f")

    print(flops,params)
