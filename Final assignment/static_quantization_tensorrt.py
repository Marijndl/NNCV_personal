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

def print_size_of_model_tensorrt(model):
    mto.save(model, "temp.pth")
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
    scripted_float_model_file = 'unet_quantization_scripted_tensorrt.pth'
    scripted_quantized_model_file = 'unet_quantization_scripted_quantized_tensorrt.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_batch_size = 32
    eval_batch_size = args.batch_size
    # Load the dataset and make a split for training and validation
    # Define the transforms to apply to the data
    train_dataloader, valid_dataloader = get_dataloaders(args)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    float_model = load_model(saved_model_dir + float_model_file, quantize=False).to(device)

    ########## Benchmark original model ###########

    print("Size of baseline model")
    print_size_of_model(float_model)

    num_eval_batches = 64
    float_model = float_model.to(device)
    dice_avg_float = evaluate(float_model, criterion, valid_dataloader, device=device, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches * eval_batch_size} images, dice: {dice_avg_float}')

    benchmark_model(float_model, valid_dataloader, device)

    ########## Quantize ###########

    config = mtq.INT8_DEFAULT_CFG

    # Forward loop
    def forward_loop(model):
        for image, target in valid_dataloader:
            target = convert_to_train_id(target)  # Convert class IDs to train IDs
            image, target = image.to(device), target.to(device)

            target = target.long().squeeze(1)  # Remove channel dimension
            output = model(image)

    # Quantize the model and perform calibration (PTQ)
    optimized_model = mtq.quantize(float_model, config, forward_loop)

    # Print quantization summary after successfully quantizing the model with mtq.quantize
    # This will show the quantizers inserted in the model and their configurations
    mtq.print_quant_summary(optimized_model)

    ########## Benchmark quantized model ###########

    print("Size of quantized model")
    print_size_of_model_tensorrt(optimized_model)

    optimized_model = optimized_model.to(device)
    dice_avg_quant = evaluate(optimized_model, criterion, valid_dataloader, device=device, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches * eval_batch_size} images, dice: {dice_avg_quant}')
    print(f'Quantization resulted in a drop of {dice_avg_quant - dice_avg_float} Dice score, which is {(dice_avg_quant - dice_avg_float)/dice_avg_float*100} % of the float model performance.')

    benchmark_model(optimized_model, valid_dataloader, device)

    ########## Save the optimized model ###########

    torch.save(mto.modelopt_state(optimized_model), saved_model_dir + "modelopt_state.pth")
    torch.save(optimized_model.state_dict(), saved_model_dir + "modelopt_weights.pth")

    print("Quantized model and successfully saved to disk")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)