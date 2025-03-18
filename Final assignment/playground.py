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

# Load pre-trained weights
weights_path = os.path.join(os.path.dirname(__file__), "unet_carvana_1.pth")
state_dict = torch.load(weights_path, map_location=device)

# Filter out decoder weights (assuming decoder layers start with "up" or "outc")
encoder_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("up", "outc"))}

# Define model and load encoder weights
num_classes_new = 19  # Update number of classes
model = UNet().to(device)
model.load_state_dict(encoder_state_dict, strict=False)  # Load only encoder weights

# Reinitialize decoder layers using Xavier initialization
def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply the initialization to decoder layers
for name, module in model.named_modules():
    if name.startswith(("up", "outc")):  # Apply to decoder layers
        module.apply(initialize_weights)


# Print layers to verify
for name, param in model.named_parameters():
    print(name, param.shape)


# Freeze encoder layers
for name, param in model.named_parameters():
    if not name.startswith(("up", "outc")):  # Encoder layers
        param.requires_grad = False  # Freeze encoder

# Only optimize decoder layers
decoder_params = [param for name, param in model.named_parameters() if name.startswith(("up", "outc"))]

# Define optimizer (only for decoder)
optimizer = torch.optim.AdamW(decoder_params, lr=1e-4, weight_decay=1e-4)

# Check which parameters are being updated
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Verify CUDA
print(next(model.parameters()).is_cuda)