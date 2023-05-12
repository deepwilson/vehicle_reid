import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
__all__ = ["seresnet18"]


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        
    def forward(self, x):
        avg_out = self.global_avgpool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1(avg_out)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2(avg_out)
        
        max_out = self.global_maxpool(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = self.fc1(max_out)
        max_out = F.relu(max_out)
        max_out = self.fc2(max_out)

        out = torch.sigmoid(avg_out + max_out)
        out = out.view(out.size(0), -1, 1, 1)
        out = x * out
        
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        
        out = x * out
        
        return out


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class CBAM_ResNet18(nn.Module):
    def __init__(self, loss, num_classes=576, pretrained=True, **kwargs):
        super(CBAM_ResNet18, self).__init__()
        self.resnet18 = torchvision.models.get_model('resnet18', pretrained=pretrained)
        
        # Add CBAM blocks to ResNet-18
        self.cbam_block1 = CBAMBlock(64)
        self.cbam_block2 = CBAMBlock(128)
        self.cbam_block3 = CBAMBlock(256)
        self.cbam_block4 = CBAMBlock(512)

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
                
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.cbam_block1(x)
        
        x = self.resnet18.layer2(x)
        x = self.cbam_block2(x)
        
        x = self.resnet18.layer3(x)
        x = self.cbam_block3(x)
        
        x = self.resnet18.layer4(x)
        x = self.cbam_block4(x)
        
        x = self.resnet18.avgpool(x)
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



def resnet18_cbam(num_classes, loss={"xent"}, **kwargs):
    model = CBAM_ResNet18(loss, num_classes, **kwargs)
    return model


if __name__=="__main__":
    from torchsummary import summary
    
    # Create an instance of SE_ResNet18 with pre-trained weights
    model = CBAM_ResNet18(pretrained=True, loss={"xent", "htri"})
    # print(summary(model, (3,224,224), device='cpu'))
    x = torch.randn(2,3,224,224)
    result = model(x)
    
    if model.loss == {"xent", "htri"}:
        print(result[0].shape, result[1].shape)
    elif model.loss == {"xent"}:
        print(result.shape)