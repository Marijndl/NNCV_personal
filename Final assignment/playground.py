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

model = smp.Unet('resnet34', encoder_weights='imagenet')

# After training your model, save it to a directory
model.save_pretrained('./my_model')

