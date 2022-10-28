# Copyright (c) OpenMMLab. All rights reserved.
from .c3d import C3D
from .potion import PoTion
from .resnet import ResNet
from .resnet3d import ResNet3d
from .resnet3d_slowfast import ResNet3dSlowFast
from .resnet3d_slowonly import ResNet3dSlowOnly
from .x3d import X3D

from .clipnet import BkCLIP, TextCLIP

__all__ = [
    'C3D', 'X3D', 'ResNet', 'ResNet3d', 'ResNet3dSlowFast', 'ResNet3dSlowOnly', 'PoTion', 'BkCLIP', 'TextCLIP'
]