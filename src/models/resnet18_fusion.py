import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ["seresnet18"]


class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        
    def forward(self, x):
        # Global average pooling
        out = self.global_avgpool(x)
        out = out.view(out.size(0), -1)
        
        # Squeeze operation
        out = self.fc1(out)
        out = F.relu(out)
        
        # Excitation operation
        out = self.fc2(out)
        out = torch.sigmoid(out)
        
        # Scale the original features
        out = out.view(out.size(0), -1, 1, 1)
        out = x * out
        
        return out

import torchvision
class SE_ResNet18(nn.Module):
    def __init__(self, loss, num_classes=576, pretrained=True, **kwargs):
        super(SE_ResNet18, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=pretrained)
        self.resnet18 = torchvision.models.get_model('resnet18', pretrained=pretrained)
        self.resnet18.layer3[0].conv2.stride = (1,1)  # Change stride of 3rd stage conv2 to 1
        self.resnet18.layer4[0].conv2.stride = (1,1)  # Change stride of 4th stage conv2 to 1
        
        # Add Squeeze and Excite blocks to ResNet-18
        self.se_block1 = SqueezeExciteBlock(64)
        self.se_block2 = SqueezeExciteBlock(128)
        self.se_block3 = SqueezeExciteBlock(256)
        self.se_block4 = SqueezeExciteBlock(512)

        self.loss = loss
        
        """Last block custom"""
        layers = []
        dropout_p = None
        hidden_dim = 1024
        layers.append(nn.Linear(512, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        self.embedding = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.embedding_fusion_block = nn.Sequential(
                                                    nn.Linear(1024, 1024),
                                                    nn.BatchNorm1d(1024),
                                                    nn.ReLU(inplace=True)
                                                )
        self.upsample = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm2d(512))
        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.se_block1(x)
        
        x = self.resnet18.layer2(x)
        x = self.se_block2(x)
        
        x = self.resnet18.layer3(x)
        x = self.se_block3(x)
        x3_cnn_out = self.upsample(x)
        #print("%%*******", x.shape)

        x3_embedding = self.resnet18.avgpool(x3_cnn_out)
        x3_embedding = x3_embedding.view(x3_embedding.size(0), -1)
        #print("***x3_embedding****", x3_embedding.shape)

        
        x = self.resnet18.layer4(x)
        x = self.se_block4(x)
        x4_embedding = self.resnet18.avgpool(x)
        x4_embedding = x4_embedding.view(x4_embedding.size(0), -1)
        #print("***x4_embedding****", x4_embedding.shape)

        embedding_fusion_block = torch.cat((x3_embedding, x4_embedding), dim=1)
        #print("***embedding_fusion_block****", embedding_fusion_block.shape)
        embedding_fusion_block = self.embedding_fusion_block(embedding_fusion_block)
        #print("*******", x.shape)

        # x = torch.cat((x, x3), dim=1)
                
        x = self.resnet18.avgpool(x)
        x = x.view(x.size(0), -1)

        v = self.embedding(x)
        # add embedding_fusion_block
        v = v + embedding_fusion_block
        
        # v = self.model(x)
        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def seresnet18_fusion(num_classes, loss={"xent"}, **kwargs):
    model = SE_ResNet18(loss, num_classes, **kwargs)
    return model

if __name__=="__main__":
    # Create an instance of SE_ResNet18 with pre-trained weights
    model = SE_ResNet18(pretrained=True, loss={"xent", "htri"})
    x = torch.randn(2,3,224,224)
    result = model(x)
    
    if model.loss == {"xent", "htri"}:
        print(result[0].shape, result[1].shape)
    elif model.loss == {"xent"}:
        print(result.shape)