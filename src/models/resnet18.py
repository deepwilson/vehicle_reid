import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ["seresnet18"]


import torchvision
class SE_ResNet18(nn.Module):
    def __init__(self, loss, num_classes=576, pretrained=True, **kwargs):
        super(SE_ResNet18, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=pretrained)
        self.resnet18 = torchvision.models.get_model('resnet18', pretrained=pretrained)
        
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
        
        x = self.resnet18.layer2(x)
        
        x = self.resnet18.layer3(x)
        
        x = self.resnet18.layer4(x)
        
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


def resnet18(num_classes, loss={"xent"}, **kwargs):
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