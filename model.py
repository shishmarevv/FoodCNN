from torchvision.models import (
    alexnet,
    resnet34,
    mobilenet_v3_large,

    AlexNet_Weights,
    ResNet34_Weights,
    MobileNet_V3_Large_Weights
)

import torch.nn as nn

class Architecture:
    input_dim : int
    output_dim : int

    def __init__(self, input_dim : int, output_dim : int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def validate(self):
        assert self.input_dim > 0, "Input dimension must be greater than 0"
        assert self.output_dim > 0, "Output dimension must be greater than 0"


class Model(nn.Module):
    def __init__(self, arch: Architecture, model_type: str = 'alexnet', pretrained: bool = False):
        super(Model, self).__init__()
        arch.validate()
        if model_type == 'alexnet':
            self.net = alexnet(weights=AlexNet_Weights.DEFAULT if pretrained else None)
            self.net.classifier[6] = nn.Linear(4096, arch.output_dim)
        elif model_type == 'resnet34':
            self.net = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
            self.net.fc = nn.Linear(512, arch.output_dim)
        elif model_type == 'mobilenet_v3_large':
            self.net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            self.net.classifier[3] = nn.Linear(1280, arch.output_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x):
        return self.net(x)
