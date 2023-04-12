# import os
# import torch
# import yaml
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.nn import Parameter
# from torch.nn import init
# import math

# def L2Normalization(ff, dim = 1):
#      # ff is B*N
#      fnorm = torch.norm(ff, p=2, dim=dim, keepdim=True) + 1e-5
#      ff = ff.div(fnorm.expand_as(ff))
#      return ff

# #https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py#L80
# class ArcLinear(nn.Module):
#     def __init__(self, in_features, out_features, s=64.0):
#         super(ArcLinear, self).__init__()
#         self.weight = Parameter(torch.Tensor(in_features,out_features))
#         init.normal_(self.weight.data, std=0.001)
#         self.loss_s = s

#     def forward(self, input):
#         embedding = input
#         nembedding = L2Normalization(embedding, dim=1)*self.loss_s
#         _weight = L2Normalization(self.weight, dim=0)
#         fc7 = nembedding.mm(_weight)
#         output = (fc7, _weight, nembedding)
#         return output

# class ArcLoss(nn.Module):
#     def __init__(self, m1=1.0, m2=0.5, m3 =0.0, s = 15.0):
#         super(ArcLoss, self).__init__()
#         self.loss_m1 = m1
#         self.loss_m2 = m2
#         self.loss_m3 = m3
#         self.loss_s = s

#     def forward(self, input, target):
#         # print("----------------->",input.shape, target.shape)
#         fc7 = input

#         index = fc7.data * 0.0 #size=(B,Classnum)
#         index.scatter_(1,target.data.view(-1,1),1)
#         index = index.byte()
#         index = Variable(index)

#         zy = fc7[index]
#         cos_t = zy/self.loss_s
#         t = torch.acos(cos_t)
#         t = t*self.loss_m1 + self.loss_m2
#         body = torch.cos(t) - self.loss_m3

#         new_zy = body*self.loss_s
#         diff = new_zy - zy
#         fc7[index] += diff
#         loss = F.cross_entropy(fc7, target)
#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
class ArcLoss(nn.Module): 
    def __init__(self, m=0.5, s=10, easy_margin=False, emb_size=128,num_classes=576):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size)).cuda()
        # num_classes 训练集中总的人脸分类数
        # emb_size 特征向量长度
        nn.init.xavier_uniform_(self.weight)
        # 使用均匀分布来初始化weight

        self.easy_margin = easy_margin
        self.m = m
        # 夹角差值 0.5 公式中的m
        self.s = s
        # 半径 64 公式中的s
        # 二者大小都是论文中推荐值

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # 差值的cos和sin
        self.th = math.cos(math.pi - self.m)
        # 阈值，避免theta + m >= pi
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        label =label.cuda()
        input =input.cuda()
        x = F.normalize(input)

        W = F.normalize(self.weight)#.t()
        # print(W.size())
        # 正则化
        cosine = (F.linear(x, W)).cuda()
        # cos值
        sine = torch.sqrt(1.0 - torch.pow(cosine/self.s, 2))
        # sin

        phi = (cosine * self.cos_m - sine * self.sin_m).cuda()
        # cos(theta + m) 余弦公式
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
            # 如果使用easy_margin
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size()).cuda()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # print(one_hot)
        # 将样本的标签映射为one hot形式 例如N个标签，映射为（N，num_classes）
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # print(output)
        # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
        # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, phi, cosine, cosine]
        # 再乘以半径，经过交叉熵，正好是ArcFace的公式
        output *= self.s
        loss = F.cross_entropy(cosine,label)

        # 乘以半径
        # return output, loss
        return loss
    



class ArcNet(nn.Module):
    def __init__(self,num_classes=576,latent_dim=128,s=20,m=0.1): # orig, s=10,m=0.1
        super().__init__()
        self.s = s # scale
        self.m = torch.tensor(m) # margin
        # self.w=nn.Parameter(torch.rand(latent_dim,num_classes)) #2*10
        self.w=nn.Parameter(torch.FloatTensor(num_classes, latent_dim)).cuda() #2*10
        nn.init.xavier_uniform_(self.w)
        self.loss = nn.NLLLoss(reduction="mean").cuda() 

    def forward(self, input, label):
        label =label.cuda()
        input =input.cuda()
        embedding = F.normalize(input,dim=1) # normalize latent output
        w = F.normalize(self.w,dim=1).cuda() # normalize weights of ArcCos network
        # w = w.cuda()
        # cos_theta = torch.matmul(embedding, w)/self.s # /10
        cos_theta = F.linear(embedding, w)/self.s # /10
        sin_theta = torch.sqrt(1.0-torch.pow(cos_theta,2))
        cos_theta_m = cos_theta*torch.cos(self.m) - sin_theta*torch.sin(self.m)
        cos_theta_scaled = torch.exp(cos_theta * self.s)
        sum_cos_theta = torch.sum(torch.exp(cos_theta*self.s),dim=1,keepdim=True) - cos_theta_scaled
        top = torch.exp(cos_theta_m*self.s)
        # arcout = (top/(top + sum_cos_theta))
        arcout = top
        # print(arcout)
        
        arcout = self.loss(arcout, label)
        # print(arcout)
        return arcout

    
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes=576, embedding_size=128, margin=0.5, scale=10):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        self.W = nn.Parameter(torch.Tensor(num_classes, embedding_size)).cuda()
        nn.init.xavier_normal_(self.W)

    def forward(self, embeddings, labels):
        embeddings, labels =  embeddings.cuda(), labels.cuda()
        batch_size = labels.size(0)

        # Compute the cosine similarity between the embeddings and the weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))

        # Compute the target mask
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device).cuda()
        onehot.scatter_(1, labels.unsqueeze(-1), 1)

        # Compute the modified cosine of target classes
        eps = 1e-6
        cosine_of_target_classes = cosine[onehot == 1]
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        modified_cosine_of_target_classes = torch.cos(angles + self.margin)

        # Compute the diff
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)

        # Compute the logits
        logits = cosine + (onehot * diff)

        # Scale the logits
        logits = logits * self.scale
        # return logits

        # Compute the cross entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss

if __name__=="__main__":
    # Define hyperparameters
    num_classes = 576
    embedding_size = 128
    margin = 0.5
    scale = 10
    
    # Create dummy data
    batch_size = 8
    embeddings = torch.randn((batch_size, embedding_size))
    labels = torch.randint(low=0, high=num_classes, size=(batch_size,))
    
    # Initialize ArcFaceLoss
    arcface_loss = ArcFaceLoss(num_classes, embedding_size, margin, scale)
    
    # Test forward pass
    loss = arcface_loss(embeddings, labels)
    print(loss)