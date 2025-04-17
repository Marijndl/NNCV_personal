from argparse import ArgumentParser

import torch.nn as nn
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
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from torch_tensorrt.ts.ptq import DataLoaderCalibrator, CalibrationAlgo
import torch_tensorrt

def print_size_of_model_tensorrt(model):
    torch.save(model.state_dict(), "temp.pth")
    print('Size (MB):', os.path.getsize("temp.pth")/1e6)
    os.remove('temp.pth')


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser

def main(args):
    saved_model_dir = './quant_models/'
    float_model_file = 'unet_float.pth'
    scripted_float_model_file = 'unet_quantization_scripted_tensorrt2.pth'
    scripted_quantized_model_file = 'unet_quantization_scripted_quantized_tensorrt2.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    train_batch_size = 32
    eval_batch_size = args.batch_size
    # Load the dataset and make a split for training and validation
    # Define the transforms to apply to the data
    train_dataloader, valid_dataloader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    float_model = load_model(saved_model_dir + float_model_file, quantize=False).to(device)
    float_model.eval()

    # Get a sample input batch for TensorRT conversion
    sample_batch, _ = next(iter(valid_dataloader))
    sample_input = sample_batch.to(device)

    # Benchmark original model
    print("Benchmarking PyTorch model:")
    benchmark_model(float_model, valid_dataloader, device)

    # Debug: Verify DataLoaderCalibrator availability and arguments
    print("Creating DataLoaderCalibrator with:")
    print(f"  dataloader: {valid_dataloader}")
    print(f"  algo_type: {CalibrationAlgo.MINMAX_CALIBRATION}")
    print(f"  cache_file: calibration.cache")
    print(f"  use_cache: False")
    print(f"  device: {device}")

    # Set up INT8 calibrator with keyword arguments as per docstring
    try:
        calibrator = DataLoaderCalibrator(
            valid_dataloader,  # Positional argument
            algo_type=CalibrationAlgo.MINMAX_CALIBRATION,
            cache_file="calibration.cache",
            use_cache=False,
            device=device
        )
        print("Calibrator created successfully.")
    except Exception as e:
        print(f"Failed to create calibrator: {e}")
        raise

    # Convert to TensorRT with INT8 precision
    trt_model = torch_tensorrt.compile(
        float_model,
        inputs=[torch_tensorrt.Input(sample_input.shape)],
        enabled_precisions={torch.int8},  # Request INT8 precision
        calibrator=calibrator,
        device=device
    )

    # Benchmark TensorRT model
    print("\nBenchmarking TensorRT INT8 model:")
    benchmark_model(trt_model, valid_dataloader, device)

    # Save the TensorRT model (optional)
    torch.jit.save(trt_model, saved_model_dir + "unet_trt_int8.ts")

    # Example inference
    with torch.no_grad():
        output = trt_model(sample_input)
    print(f"Sample output shape: {output.shape}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)