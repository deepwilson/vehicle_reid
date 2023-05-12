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
        
        # Add Squeeze and Excite blocks to ResNet-18
        self.se_block1 = SqueezeExciteBlock(64)
        self.se_block2 = SqueezeExciteBlock(128)
        self.se_block3 = SqueezeExciteBlock(256)
        self.se_block4 = SqueezeExciteBlock(512)

        self.loss = loss
        
        """Last block custom"""
        layers = []
        dropout_p = None
        hidden_dim = 128
        layers.append(nn.Linear(512, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        self.embedding = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.se_block1(x)
        
        x = self.resnet18.layer2(x)
        x = self.se_block2(x)
        
        x_3 = self.resnet18.layer3(x)
        x_3 = self.se_block3(x_3)
        
        x_4 = self.resnet18.layer4(x_3)
        x_4 = self.se_block4(x_4)
        
        # Fuse the features from the 3rd and 4th stage
        x_fused = torch.cat((x_3, x_4), dim=1)

        x = self.resnet18.avgpool(x_fused)
        x = x.view(x.size(0), -1)

        v = self.embedding(x)
        
        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")

def seresnet18(num_classes, loss={"xent"}, **kwargs):
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