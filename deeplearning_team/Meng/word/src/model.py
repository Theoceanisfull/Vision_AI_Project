import torch.nn as nn
import torchvision.models.video as vmodels


def build_model(backbone: str, num_classes: int, dropout: float = 0.5):
    """Build a pretrained video backbone with a new classification head.

    Supported backbones (all pretrained on Kinetics-400):
      r2plus1d_18  — R(2+1)D, best accuracy/speed trade-off (recommended)
      mc3_18       — Mixed Convolution 3D, slightly faster
      r3d_18       — Pure 3D ResNet-18, fastest

    The final FC is replaced with Dropout + Linear(in_features, num_classes).
    """
    if backbone == "r2plus1d_18":
        weights = vmodels.R2Plus1D_18_Weights.KINETICS400_V1
        model = vmodels.r2plus1d_18(weights=weights)
    elif backbone == "mc3_18":
        weights = vmodels.MC3_18_Weights.KINETICS400_V1
        model = vmodels.mc3_18(weights=weights)
    elif backbone == "r3d_18":
        weights = vmodels.R3D_18_Weights.KINETICS400_V1
        model = vmodels.r3d_18(weights=weights)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_param_groups(model, base_lr: float, head_lr_multiplier: float = 10.0):
    """Return optimizer param groups with differential learning rates.

    Backbone layers get base_lr; the new head gets base_lr * head_lr_multiplier.
    This is essential because the head is randomly initialized while the backbone
    already has good Kinetics features.
    """
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
    head_params     = [p for n, p in model.named_parameters() if     n.startswith('fc')]
    return [
        {'params': backbone_params, 'lr': base_lr},
        {'params': head_params,     'lr': base_lr * head_lr_multiplier},
    ]
