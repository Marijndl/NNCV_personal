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



def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
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

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    float_model = load_model(saved_model_dir + float_model_file, quantize=False).to(device)

    ########## Tutorial part ###########

    config = mtq.INT8_DEFAULT_CFG

    # Define forward_loop. Please wrap the data loader in the forward_loop
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

    # Save the optimized model
    torch.save(mto.modelopt_state(optimized_model), saved_model_dir + "modelopt_state.pth")
    torch.save(optimized_model.state_dict(), saved_model_dir + "modelopt_weights.pth")

    torch.jit.save(torch.jit.script(optimized_model), saved_model_dir + scripted_quantized_model_file)
    print("Quantized model and successfully saved to disk")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)