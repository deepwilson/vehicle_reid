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
        layers.append(nn.Linear(512*4, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        # if dropout_p is not None:
        #     layers.append(nn.Dropout(p=dropout_p))
        self.embedding = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        # self.embedding_fusion_block = nn.Sequential(
        #                                             nn.Linear(1024, 1024),
        #                                             nn.BatchNorm1d(1024),
        #                                             nn.ReLU(inplace=True)
        #                                         )
        self.modify_stage3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                                    #   nn.BatchNorm2d(512)
                                      )
        self.modify_stage2 = nn.Sequential(nn.Conv2d(128, 512, kernel_size=1, stride=4, bias=False),
                                    #   nn.BatchNorm2d(512)
                                      )
        
        self.conv_downsample_stage3 = nn.Sequential(
                                            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True)
                                        )
        self.conv_downsample_stage2 = nn.Sequential(
                                            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True)
                                        )
        self.conv_downsample_stage1 = nn.Sequential(
                                            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True)
                                        )

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x_stage1 = self.conv_downsample_stage1(x)
        x = self.resnet18.layer1(x)
        x = self.se_block1(x)

        #print("stage1 shape", x.shape)
        #print("stage1 modified shape", x_stage1.shape)
        
        x = self.resnet18.layer2(x)
        x = self.se_block2(x)
        x_stage2 = self.conv_downsample_stage2(x)
        #print("stage2 shape", x.shape)
        #print("stage2 modified shape", x_stage2.shape)
        
        x = self.resnet18.layer3(x)
        x = self.se_block3(x)
        
        x_stage3 = self.modify_stage3(x)
        #print("stage3 shape", x.shape)
        #print("stage3 modified shape", x_stage3.shape)
        
        x = self.resnet18.layer4(x)
        x = self.se_block4(x)

        # x = torch.cat((x, x3), dim=1)
        # x = x + x_

                
        x = torch.cat((x, x_stage3, x_stage2, x_stage1), dim=1)
        #print("stage3 shape", x.shape)
        x = self.resnet18.avgpool(x)
        x = x.view(x.size(0), -1)

        v = self.embedding(x)
        
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


def seresnet18_concatfusion(num_classes, loss={"xent"}, **kwargs):
    model = SE_ResNet18(loss, num_classes, **kwargs)
    return model


if __name__=="__main__":
    from torchsummary import summary
    
    # Create an instance of SE_ResNet18 with pre-trained weights
    model = SE_ResNet18(pretrained=True, loss={"xent", "htri"})
    #print(summary(model, (3,224,224), device='cpu'))
    x = torch.randn(2,3,224,224)
    result = model(x)
    
    if model.loss == {"xent", "htri"}:
        print(result[0].shape, result[1].shape)
    elif model.loss == {"xent"}:
        print(result.shape)