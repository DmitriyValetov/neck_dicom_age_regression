import torch
from efficientnet_pytorch import EfficientNet



class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class LayerReplacer(torch.nn.Module):
    def forward(self, x):
        return x


def make_efficientnet(num_classes, device='cuda', i=0):
    model = EfficientNet.from_pretrained(f'efficientnet-b{i}')
    model._conv_stem = torch.nn.Conv2d(1, 32, (3,3), (2,2), bias=False)
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(num_ftrs, 1000, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(1000, num_classes, bias = True)
    )
    model.to(device)
    return model


def make_efficientnet_featurizer(device='cuda', i=0):
    model = EfficientNet.from_pretrained(f'efficientnet-b{i}')
    model._conv_stem = torch.nn.Conv2d(1, 32, (3,3), (2,2), bias=False)
    model._fc = LayerReplacer()
    model._swish = LayerReplacer()
    model.to(device)
    return model