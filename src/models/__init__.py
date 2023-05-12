# Copyright (c) EEEM071, University of Surrey

from .resnet import resnet50, resnet50_fc512
from .seresnet import seresnext50
from .seresnet18 import seresnet18
from .resnet18 import resnet18
from .resnet18_cbam import resnet18_cbam
from .resnet18_cbam_concatfusion import resnet18_cbam_concatfusion
from .seresnet18_additionfusion import seresnet18_additionfusion
from .seresnet18_concatfusion import seresnet18_concatfusion
# from .seresnet18fusion import seresnet18_fusion
from .mobilenet import mobilenet_v3_small
from .samobilenet import samobilenet_v3_small




__model_factory = {
    # image classification models
    "resnet50": resnet50,
    "seresnext50": seresnext50,
    "resnet18": resnet18,
    "seresnet18": seresnet18,
    "resnet18_cbam": resnet18_cbam,
    "resnet18_cbam_concatfusion": resnet18_cbam_concatfusion,
    "seresnet18_additionfusion": seresnet18_additionfusion,
    "seresnet18_concatfusion": seresnet18_concatfusion,
    "mobilenet_v3_small": mobilenet_v3_small,
    "resnet50_fc512": resnet50_fc512,
    "samobilenet_v3_small": samobilenet_v3_small,

}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)