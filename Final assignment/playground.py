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
from vltseg.models.eva02 import EVA02

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained weights
teacher_model = EVA02().to(device)
checkpoint = torch.load('vltseg_checkpoint_mapillary+cityscapes_2.pth')
teacher_model.load_state_dict(checkpoint['model_state_dict'])
teacher_model.eval()  # Set the model to evaluation mode

