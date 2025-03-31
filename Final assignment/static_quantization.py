import os
import sys
import time
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.ao.quantization import QuantStub, DeQuantStub

import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,

)
from utils import *

from unet import UNet
import segmentation_models_pytorch as smp

from utils import *


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="D:\Cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--decoder", type=str, default="resnext101_32x8d", help="Decoder name for the DeepLabV3+ model")

    return parser


def main(args):
    saved_model_dir = r'C:\Users\20203226\Documents\GitHub\NNCV\Final assignment\models'
    float_model_file = r'\unet_float.pth'
    scripted_float_model_file = 'unet_quantization_scripted.pth'
    scripted_quantized_model_file = 'unet_quantization_scripted_quantized.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_batch_size = 32
    eval_batch_size = args.batch_size
    # Load the dataset and make a split for training and validation
    # Define the transforms to apply to the data
    transform = Compose([
        ToImage(),
        Resize((256, 256), antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize((0.2854, 0.3227, 0.2819), (0.04797, 0.04296, 0.04188)),
    ])

    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic",
                               transforms=transform, )
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform, )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # pin_memory=True, persistent_workers=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # pin_memory=True, persistent_workers=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    float_model = load_model(saved_model_dir + float_model_file, quantize=False).to('cpu')

    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    print('\n Inc Block: Before fusion \n\n', float_model.inc.double_conv)
    float_model.eval()

    print("--------------------------------------")

    # for name, module in float_model.named_modules():
    #     print(name, type(module))

    # Fuses modules
    float_model.to('cpu')
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inc Block: After fusion\n\n', float_model.inc.double_conv)

    print("Size of baseline model")
    print_size_of_model(float_model)

    num_eval_batches = 64

    float_model = float_model.to(device)
    dice_avg = evaluate(float_model, criterion, valid_dataloader, device=device, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches * eval_batch_size} images, dice: {dice_avg}')
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

    # Fine tune model on training data
    per_channel_quantized_model = load_model(saved_model_dir + float_model_file, quantize=True)
    per_channel_quantized_model = per_channel_quantized_model.to('cpu')
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print(per_channel_quantized_model.qconfig)

    num_calibration_batches = 20

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model,criterion, train_dataloader, num_calibration_batches)
    torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)

    # Evaluation after quantization:
    print("Size of model after quantization")
    print_size_of_model(per_channel_quantized_model)

    dice_avg = evaluate(per_channel_quantized_model, criterion, valid_dataloader, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches * eval_batch_size} images, dice: {dice_avg}')
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
