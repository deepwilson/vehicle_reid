import torch.nn as nn
import torchvision.models as tvmodels
import torch
from torchsummary import summary
__all__ = ["mobilenet_v3_small", "vgg16"]


class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, **kwargs):
        super().__init__()

        self.loss = loss
        self.backbone = tvmodels.__dict__[name](pretrained=pretrained)
        self.feature_dim = self.backbone.classifier[0].in_features

        # overwrite the classifier used for ImageNet pretrianing
        # nn.Identity() will do nothing, it's just a place-holder

        


        # Vanilla
        # self.backbone.classifier = nn.Identity()
        # self.classifier = nn.Linear(self.feature_dim, num_classes)

        # """Last block custom FC-ReLU-Dropout-BN-FC"""
        # layers = []
        # dropout_p = 0.5
        # hidden_dim = 1024
        # layers.append(nn.Linear(self.feature_dim, hidden_dim))
        # layers.append(nn.ReLU(inplace=True))
        # if dropout_p is not None:
        #     layers.append(nn.Dropout(p=dropout_p))
        # layers.append(nn.BatchNorm1d(hidden_dim))
        # self.backbone.classifier = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_dim, num_classes)

        # """Last block custom FC-ReLU-BN-FC"""
        # layers = []
        # dropout_p = 0.5
        # hidden_dim = 1024
        # layers.append(nn.Linear(self.feature_dim, hidden_dim))
        # layers.append(nn.ReLU(inplace=True))
        # # if dropout_p is not None:
        # #     layers.append(nn.Dropout(p=dropout_p))
        # layers.append(nn.BatchNorm1d(hidden_dim))
        # self.backbone.classifier = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_dim, num_classes)

        
        # """block custom FC-BN-ReLU-Dropout-FC"""
        # layers = []
        # dropout_p = 0.5
        # hidden_dim = 1024
        # layers.append(nn.Linear(self.feature_dim, hidden_dim))
        # layers.append(nn.BatchNorm1d(hidden_dim))
        # layers.append(nn.ReLU(inplace=True))
        # if dropout_p is not None:
        #     layers.append(nn.Dropout(p=dropout_p))
        # self.backbone.classifier = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_dim, num_classes)


        # """block custom FC-BN-ReLU-FC"""
        # layers = []
        # dropout_p = 0.5
        # hidden_dim = 1024
        # layers.append(nn.Linear(self.feature_dim, hidden_dim))
        # layers.append(nn.BatchNorm1d(hidden_dim))
        # layers.append(nn.ReLU(inplace=True))
        # # if dropout_p is not None:
        # #     layers.append(nn.Dropout(p=dropout_p))
        # self.backbone.classifier = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_dim, num_classes)

        """block custom FC-BN-FC"""
        layers = []
        dropout_p = 0.5
        hidden_dim = 1024
        layers.append(nn.Linear(self.feature_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        # layers.append(nn.ReLU(inplace=True))
        # if dropout_p is not None:
        #     layers.append(nn.Dropout(p=dropout_p))
        self.backbone.classifier = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        v = self.backbone(x)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")
        
def mobilenet_v3_small(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

if __name__=="__main__":
    net = mobilenet_v3_small(576, loss={"xent", "htri"})
    x = torch.randn(4,3,224,224)
    # net.eval()
    # with torch.no_grad():
    #     out = net(x)
    print(summary(net, (3,224,224), device="cpu"))
    out = net(x)
    print("$$$$$", out[0].shape, out[1].shape)
    # print(out[1].shape)