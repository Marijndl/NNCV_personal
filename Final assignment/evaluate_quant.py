import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)
from torch.profiler import profile, ProfilerActivity

from utils import *
from fvcore.nn import FlopCountAnalysis


def calculate_flops(model, images, device):
    model = model.to(device)
    model.eval()
    images = images.to(device)

    try:
        flop_counter = FlopCountAnalysis(model, images)
        return flop_counter.total()
    except Exception as e:
        print(f"Error calculating FLOPS: {e}")
        return 0


def run_benchmark(model_file, img_loader, device):
    print(f'Running {model_file}')
    elapsed = 0
    try:
        if model_file.endswith('unet_float.pth'):
            model = load_model(model_file, False)
        else:
            model = torch.jit.load(model_file)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)
    model.eval()
    num_batches = 20
    flops = 0

    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        images, target = images.to(device), target.to(device)
        if i == 0:
            flops = calculate_flops(model, images, device)

        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed += (end - start)
        else:
            break

    num_images = images.detach().size()[0] * num_batches
    print(f'Elapsed time: {elapsed / num_images * 1000:.3f} ms')
    print(f'Estimated FLOPS: {flops:.6f}')
    return elapsed


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="D:\Cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


def main(args):
    saved_model_dir = r'models'
    float_model_file = r'\unet_float.pth'
    scripted_float_model_file = 'unet_quantization_scripted.pth'
    scripted_quantized_model_file = 'unet_quantization_scripted_quantized.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset and make a split for training and validation
    # Define the transforms to apply to the data
    transform = Compose([
        ToImage(),
        Resize((256, 256), antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize((0.2854, 0.3227, 0.2819), (0.04797, 0.04296, 0.04188)),
    ])

    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_benchmark(saved_model_dir + scripted_quantized_model_file, valid_dataloader, device='cpu')
    run_benchmark(r"models/unet_float.pth", valid_dataloader, device='cpu')
    run_benchmark(saved_model_dir + scripted_float_model_file, valid_dataloader, device='cpu')


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
