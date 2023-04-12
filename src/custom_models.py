import torch.nn as nn
from torchvision import models
import torch
# from src.custom_losses import ArcLoss


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class Modified_Head(nn.Module):
    def __init__(self, backbone_output_size=2048, bottle_neck=128, num_classes=576, dropout_prob=0.1, exclude_classifier=False):
        super().__init__()
        self.exclude_classifier = exclude_classifier

        self.fc1 = nn.Linear(backbone_output_size, bottle_neck)
        self.fc1.apply(weights_init_classifier)
        
        # Add batch normalization, ReLU activation, and dropout layers
        self.bn = nn.BatchNorm1d(bottle_neck)
        self.bn.apply(weights_init_kaiming)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Add fully connected layer for classification
        self.classifier = nn.Linear(bottle_neck, num_classes)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x):
        # print("--------------------->", x.shape)
        # Apply bottleneck layer
        x = self.fc1(x)
        # Apply batch normalization, ReLU activation, and dropout
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)   
        # Apply fully connected layer for classification
        if not self.exclude_classifier:
            x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ResNet(nn.Module):

    def __init__(self, num_classes=576):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.head = Modified_Head()
        # self.head.classifier = ArcLoss()
           
    def forward(self, x):
        # pass though resnet but don't include final fc layer(2048->1000)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1)) #(bs, num_features(2048) )
        
        # pass though modified head (2048->512?)
        x = self.head(x)
        return x
    

if __name__== "__main__":
    input = torch.FloatTensor(4, 3, 320, 320) #(bs, num_channels, height, width)
    net = ResNet(num_classes=576)
    output = net(input)
    print(output.shape)
