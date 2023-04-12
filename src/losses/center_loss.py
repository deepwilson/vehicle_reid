import torch
import torch.nn as nn

from typing import Tuple

import torch
from torch import nn, Tensor
# def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
#     similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
#     label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

#     positive_matrix = label_matrix.triu(diagonal=1)
#     negative_matrix = label_matrix.logical_not().triu(diagonal=1)

#     similarity_matrix = similarity_matrix.view(-1)
#     positive_matrix = positive_matrix.view(-1)
#     negative_matrix = negative_matrix.view(-1)
#     return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

# class CenterLoss(nn.Module):
#     """Center loss.
    
#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """
#     def __init__(self, num_classes=576, feat_dim=512, use_gpu=True):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu

#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, self.centers.t())

#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))

#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

#         return loss

class CenterLoss(nn.Module):
    """this one worked don't delete!
    Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=576, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
 
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

# from __future__ import absolute_import

# import torch
# from torch import nn

# class CenterLoss(nn.Module):
#     def __init__(self,cls_num,feature_num):
#         super().__init__()
#         self.cls_num = cls_num
#         self.center = nn.Parameter(torch.randn(cls_num,feature_num))

#     def forward(self, xs,label):
#         xs = f.normalize(xs)
#         #根据label索引选择中心点
#         cen_select = self.center.index_select(dim=0,index=label)
#         #统计出每个类的data---->[2,1]
#         count = torch.histc(label.float(),bins=10,min=0,max=9)
#         #根据count出来的数量从label里重新选择，count_dis为每个data对于的数量----》[2,2,1]
#         count_dis = count.index_select(dim=0,index=label)

#         return torch.sum(torch.sum((xs-cen_select)**2,dim =1)/count_dis.float())

# # class CenterLoss(nn.Module):
# #     """Center loss.
# #     Reference:
# #     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
# #     Args:
# #         num_classes (int): number of classes.
# #         feat_dim (int): feature dimension.
# #     """

# #     def __init__(self, num_classes=576, feat_dim=2048, use_gpu=True):
# #         super(CenterLoss, self).__init__()
# #         self.num_classes = num_classes
# #         self.feat_dim = feat_dim
# #         self.use_gpu = use_gpu

# #         if self.use_gpu:
# #             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
# #         else:
# #             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

# #     def forward(self, x, labels):
# #         """
# #         Args:
# #             x: feature matrix with shape (batch_size, feat_dim).
# #             labels: ground truth labels with shape (num_classes).
# #         """
# #         assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

# #         batch_size = x.size(0)
# #         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
# #                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
# #         distmat.addmm_(1, -2, x, self.centers.t())

# #         classes = torch.arange(self.num_classes).long()
# #         if self.use_gpu: classes = classes.cuda()
# #         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
# #         mask = labels.eq(classes.expand(batch_size, self.num_classes))

# #         dist = []
# #         for i in range(batch_size):
# #             value = distmat[i][mask[i]]
# #             value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
# #             dist.append(value)
# #         dist = torch.cat(dist)
# #         loss = dist.mean()
# #         return loss



if __name__ == '__main__':
    use_gpu = False
    center_loss = CenterLoss(576,2048,use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)
    print(loss)
