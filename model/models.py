from .DDPM import DDPM
from .DDPM_guide import DDPM_guide
from .EDM import EDM
from .unet import UNet
from .unet_guide import UNet_guide

CLASSES = {
    cls.__name__: cls
    for cls in [DDPM, DDPM_guide, EDM, UNet, UNet_guide]
}


def get_models_class(model_type='DDPM', net_type='UNet', guide=False):
    if guide:
        model_type += '_guide'
        net_type += '_guide'
    return CLASSES[model_type], CLASSES[net_type]
