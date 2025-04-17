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
import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp


def print_size_of_model_tensorrt(model):
    torch.save(model.state_dict(), "temp.pth")
    print('Size (MB):', os.path.getsize("temp.pth") / 1e6)
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
    pruned_model_file = 'unet_pruned.pth'

    device = torch.device("cpu")  # Note: pruning might work better on CPU for some operations
    print(f"Using device: {device}")

    train_batch_size = 32
    eval_batch_size = args.batch_size

    # Load dataset
    train_dataloader, valid_dataloader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
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

    ########## Prune the model ###########

    # Define pruning configuration
    config = mtp.config.FastNASConfig()

    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    # Forward loop for calibration
    def forward_loop(model):
        for image, target in valid_dataloader:
            target = convert_to_train_id(target)
            image, target = image.to(device), target.to(device)
            target = target.long().squeeze(1)
            output = model(image)
            loss = criterion(output, target)  # Using your existing loss setup

    # Wrap evaluate function as the metric
    def score_func(model):
        return evaluate(model, criterion, valid_dataloader, device=device, neval_batches=num_eval_batches)

    # prune the model
    pruned_model, _ = mtp.prune(
        model=float_model,
        mode=[("fastnas", config)],
        constraints={"flops": "75%"},
        dummy_input=dummy_input,
        config={
            "data_loader": train_dataloader,
            "score_func": score_func,
            "checkpoint": saved_model_dir + "modelopt_seaarch_checkpoint_fastnas.pth",
        },
    )

    # Print pruning summary
    mtp.print_pruning_summary(pruned_model)

    ########## Benchmark pruned model ###########

    print("Size of pruned model")
    print_size_of_model_tensorrt(pruned_model)

    pruned_model = pruned_model.to(device)
    dice_avg_pruned = evaluate(pruned_model, criterion, valid_dataloader, device=device, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches * eval_batch_size} images, dice: {dice_avg_pruned}')
    print(f'Pruning resulted in a drop of {dice_avg_float - dice_avg_pruned} Dice score, which is {(dice_avg_float - dice_avg_pruned) / dice_avg_float * 100} % of the float model performance.')

    torch.save(mto.modelopt_state(pruned_model), saved_model_dir + "modelopt_state_pruning.pth")
    torch.save(pruned_model.state_dict(), saved_model_dir + "modelopt_weights_pruning.pth")

    benchmark_model(pruned_model, valid_dataloader, device)

    ########## Save the pruned model ###########



    print("Pruned model successfully saved to disk")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)