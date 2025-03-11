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
from unet_model import UNet, OutConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model with the new number of classes
num_classes_new = 19  # Set to your new number of classes
net = UNet().to(device)  # Ensure your UNet constructor supports this

# Load pre-trained weights
weights_path = os.path.join(os.path.dirname(__file__), "unet_carvana_1.pth")
state_dict = torch.load(weights_path, map_location=device)

# Remove the last layer's weights (assuming "out.conv.weight" and "out.conv.bias" are its names)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("outc.conv")}

# Load filtered state_dict
net.load_state_dict(state_dict, strict=False)  # strict=False ignores missing keys (like out.conv)

# Reinitialize the last layer with the correct number of output classes
net.outc = OutConv(net.outc.conv.in_channels, num_classes_new)

# net.to(device)  # Move model to GPU if available

# Print layers to verify
for name, param in net.named_parameters():
    print(name, param.shape)

print(next(net.parameters()).is_cuda)