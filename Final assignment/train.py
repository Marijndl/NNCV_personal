"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype, InterpolationMode,
    RandomRotation, RandomHorizontalFlip, RandomVerticalFlip,
    RandomResizedCrop, RandomAffine
)
from utils import * 

from unet import UNet

import subprocess
import sys

package_name = "optuna"

try:
    __import__(package_name)
except ImportError:
    print(f"{package_name} not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package_name])

import optuna  # Now the package should be available


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--run-optuna", action="store_true", help="Enable Optuna hyperparameter optimization")

    return parser


def objective(trial):
    """Objective function for Optuna Bayesian optimization."""
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    scheduler_type = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    
    if scheduler_type == "StepLR":
        step_size = trial.suggest_int("step_size", 5, 30)
        gamma = trial.suggest_uniform("gamma", 0.1, 0.9)
    elif scheduler_type == "CosineAnnealingLR":
        T_max = trial.suggest_int("T_max", 10, 50)
    elif scheduler_type == "ReduceLROnPlateau":
        patience = trial.suggest_int("patience", 3, 10)
        factor = trial.suggest_uniform("factor", 0.1, 0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, n_classes=19).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience)

    transform = Compose([
        ToImage(),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR),
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        ToDtype(torch.float32, scale=True),
        Normalize((0.2869, 0.3251, 0.2839), (0.1869, 0.1901, 0.1872)),
    ])

    train_dataset = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform))
    valid_dataset = wrap_dataset_for_transforms_v2(Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    best_valid_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_dataloader:
            labels = convert_to_train_id(labels).long().squeeze(1)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_dice_scores = []
        with torch.no_grad():
            for images, labels in valid_dataloader:
                labels = convert_to_train_id(labels).long().squeeze(1)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                dice = dice_score(outputs, labels)
                valid_dice_scores.append(dice)

        avg_dice = sum(valid_dice_scores) / len(valid_dice_scores)
        best_valid_dice = max(best_valid_dice, avg_dice)

        if scheduler:
            if isinstance(scheduler, CosineAnnealingLR) or isinstance(scheduler, StepLR):
                scheduler.step()
            elif isinstance(scheduler, ReduceLROnPlateau):
                # Update the scheduler based on validation loss or any metric
                val_loss = best_valid_dice  # You need to compute validation loss here
                scheduler.step(val_loss)  # Pass the validation loss to `step()`

        trial.report(avg_dice, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_valid_dice

def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation", name=args.experiment_id, config=vars(args))

    if args.run_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials)

        print("Best Hyperparameters:", study.best_trial.params)
        return  # Exit after tuning

    # Train with default args if Optuna is not enabled
    best_params = {"lr": args.lr, "weight_decay": 1e-4, "scheduler": "CosineAnnealingLR", "T_max": args.epochs}
    print(f"Training U-Net with default params: {best_params}")

    # You can call `objective()` here with best params if you want to run training with optimal params found earlier.

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
