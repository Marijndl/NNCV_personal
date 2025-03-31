import time
from argparse import ArgumentParser

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)

from utils import *


def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end - start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed / num_images * 1000))
    return elapsed


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

    run_benchmark(saved_model_dir + scripted_float_model_file, valid_dataloader)
    run_benchmark(saved_model_dir + scripted_quantized_model_file, valid_dataloader)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)