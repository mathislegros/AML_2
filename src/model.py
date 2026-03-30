import segmentation_models_pytorch as smp
import torch

def get_model(arch='Unet', encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation=None):
    # Create model based on architecture name
    if arch == 'Unet':
        model_class = smp.Unet
    elif arch == 'UnetPlusPlus':
        model_class = smp.UnetPlusPlus
    elif arch == 'DeepLabV3Plus':
        model_class = smp.DeepLabV3Plus
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model_class(
        encoder_name=encoder_name, 
        encoder_weights=encoder_weights, 
        in_channels=3, 
        classes=classes, 
        activation=activation
    )
    return model
