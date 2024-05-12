import torch
import torch.nn as nn
import torch.nn.functional as F

from .grnet import GRNet

class SFCloss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2):
        super(SFCloss, self).__init__()
        self.num_classes = num_classes # N+1
        self.feat_dim = feat_dim
        # close_centers N,feat_dim
        self.close_numclasses = self.num_classes-1
        self.close_centers = nn.Parameter(torch.randn(self.close_numclasses, self.feat_dim))
        # open centers 1,feat_dim
        self.open_centers = nn.Parameter(torch.randn(1, self.feat_dim))
        self.out_weight = nn.Parameter(torch.zeros((1)))
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        labels_class = labels[:,0] #(0,N-1)
        labels_sim = labels[:,1]*int(self.num_classes) #(0 or N)
        batch_size = x.size(0)
        
        distmat_close = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.close_numclasses) + \
                  torch.pow(self.close_centers, 2).sum(dim=1, keepdim=True).expand(self.close_numclasses, batch_size).t()
        
        distmat_close.addmm_(x, self.close_centers.t(), beta=1, alpha=-2)
        # distmat_close:torch.Size([B, N])
        out_weight = torch.sigmoid(self.out_weight)
        mid_centers = (1-out_weight)*self.close_centers + out_weight*self.open_centers

        distmat_open = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, 1) + \
                  torch.pow(mid_centers, 2).sum(dim=1, keepdim=True).expand(self.close_numclasses, batch_size).t()
        # distmat_open += 2*x@self.centers.t()
        distmat_open.addmm_(x, mid_centers.t(), beta=1, alpha=-2)
        # distmat_open:torch.Size([B,N])

        classes = torch.arange(self.close_numclasses,dtype=torch.int64,device=x.device)

        labels_class = labels_class.unsqueeze(1).expand(batch_size, self.close_numclasses)
        
        mask = labels_class.eq(classes.expand(batch_size, self.close_numclasses)).float()
        
        dist_close = (distmat_close * mask.float()).clamp(min=1e-12, max=1e+12).sum(1) #(B)
        dist_open = (distmat_open * mask.float()).clamp(min=1e-12, max=1e+12).sum(1) #(B)
        use_syn_mask = (labels[:,1]==1) #(B)
        # mask_pred:torch.Size([B, class])
        dist = dist_close*(~use_syn_mask) + use_syn_mask*(dist_open)
        loss = dist.sum() / batch_size
        return loss

class SFCR(GRNet):
    def __init__(self, data_shape=(1,16384,2), expand=[2*4,2*4,4*4,8*4,16*4], num_classes=100,fusion=[], loss_weight=0.001):
        super(SFCR, self).__init__(data_shape=data_shape,expand=expand,num_classes=num_classes,fusion=fusion)
        feature_size = data_shape[0]*expand[-1]
        self.feat_dim = 256
        self.fc = nn.Linear(feature_size,self.feat_dim)
        self.fc1 = nn.Linear(self.feat_dim, self.num_classes)
        self.num_classes=num_classes
        self.loss = SFCloss(num_classes=num_classes,feat_dim=self.feat_dim)
        self.loss_weight =loss_weight

    def get_loss(self, out, y=None, reduction='mean'):
        # y is None for test
        (pred,feature) = out
        labels_class = y[:,0] # [0,N-1]
        labels_syn = y[:,1] # [0,1]
        label_open = labels_class.clone()
        label_open[labels_syn==1] = self.num_classes-1
        loss_center = self.loss(feature, y)
        loss_cel = F.cross_entropy(pred,label_open)
        return loss_cel+self.loss_weight *loss_center

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        feature = torch.flatten(x, 1)
        latent = self.fc(feature)
        pred = self.fc1(latent)
        return pred, latent

if __name__ == '__main__':
    center = SFCloss(11,100,0.1)
    x = torch.randn((25,100))
    labels_pred = torch.randint(0,10,(5,1))
    labels_syn = torch.randint(0,2,(5,1))
    print(f'pred:{labels_pred},syn:{labels_syn}')
    y = torch.cat((labels_pred,labels_syn),1)
    b = center(x,y)