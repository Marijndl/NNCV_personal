import os
from argparse import ArgumentParser

# import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)

from unet import UNet
import segmentation_models_pytorch as smp


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model
model = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",  
    encoder_weights="imagenet",  
    decoder_channels=512,  
    decoder_atrous_rates=(6, 12, 18),  
    in_channels=3,  
    classes=19,  
    activation=None,  
    aux_params=dict(
        pooling="avg",  
        dropout=0.2,  
        activation="softmax2d",  
        classes=19  
    )
)


from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model)
