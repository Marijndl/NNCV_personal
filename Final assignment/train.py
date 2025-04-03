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
from sipbuild.generator.parser.annotations import boolean
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

saved_models = []
max_saved_models = 3  # Keep last 3 best models

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
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--model", type=str, default="unet", help="Choose the model to train")
    parser.add_argument("--decoder", type=str, default="resnext101_32x8d", help="Decoder name for the DeepLabV3+ model")
    parser.add_argument("--motion-blur", type=bool, default=False, help="Wether to include the motion blur data augmentation")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the transforms to apply to the images
    # Define the transforms to apply to the images
    transform_train = Compose([
          ToImage(),
          RandomHorizontalFlip(0.5),
          # RandomCrop((512, 1024)),  # Crop to focus on smaller details
          Resize((512, 512), interpolation=InterpolationMode.BILINEAR, antialias=True),
          ToDtype(torch.float32, scale=True),
          Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ] + ([MotionBlurTransform()] if args.motion_blur else []))

    transform_val = Compose([
        ToImage(),
        Resize((512, 512), interpolation=InterpolationMode.BILINEAR, antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform_train,
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform_val,
    )
    test_dataset = Cityscapes(
        args.data_dir,
        split="test",
        mode="fine",
        target_type="semantic",
        transforms=transform_val,
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)
    test_dataset = wrap_dataset_for_transforms_v2(test_dataset)

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
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # pin_memory=True, persistent_workers=True
    )

    # Define the model
    if args.model == "unet":
        model = UNet
    elif args.model == "deeplab":
        model = smp.DeepLabV3Plus(
            encoder_name=args.decoder,
            encoder_weights="imagenet",
            decoder_channels=512,
            decoder_atrous_rates=(6, 12, 18),
            in_channels=3,
            classes=19,
            activation=None,
            aux_params=dict(
                pooling="avg",
                dropout=0.2,
                activation="softmax2d",
                classes=19
            )
        )
    else:
        raise ValueError(f"Model {args.model} is not supported, choose a different model")
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    # criterion_dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Compute total training steps
    total_steps = args.epochs * len(train_dataloader)

    # Define OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=int(1.5 * total_steps), pct_start=0.2)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):  # Some segmentation models return (logits, aux_output)
                outputs = outputs[0]  # Keep only the segmentation output
            # loss_dice = criterion_dice(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss.detach().item())

            # Step OneCycleLR scheduler
            scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                # "train_DICE_loss": loss_dice.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            # losses_dice = []
            losses = []
            dice_scores = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                if isinstance(outputs, tuple):  # Some segmentation models return (logits, aux_output)
                    outputs = outputs[0]  # Keep only the segmentation output
                # loss_dice = criterion_dice(outputs, labels)
                loss = criterion(outputs, labels)

                dice = dice_score(outputs, labels)
                # losses_dice.append(loss_dice.item())
                losses.append(loss.item())
                dice_scores.append(dice)
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            # valid_loss_dice = sum(losses_dice) / len(losses_dice)
            valid_loss = sum(losses) / len(losses)

            valid_dice = sum(dice_scores) / len(dice_scores)
            wandb.log({
                "valid_loss": valid_loss,
                # "valid_DICE_loss": valid_loss_dice,
                "valid_dice_score": valid_dice,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_path = os.path.join(output_dir, f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                saved_models.append(model_path)

                model.save_pretrained('./resnest101e')
                if len(saved_models) > max_saved_models:
                    os.remove(saved_models.pop(0))  # Remove the oldest model
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )

    # Evaluate the model on artificial test set.
    model.to('cpu')
    model.eval()
    print(f"Size of {args.model}, backbone: {args.backbone if args.backbone is not None else ""} model")
    print_size_of_model(model)

    num_eval_batches = 64
    dice_avg = evaluate(model, criterion, test_dataloader, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches * args.batch_size} images, dice: {dice_avg}')

    print("GPU:")
    benchmark_model(model, test_dataloader, device=device)

    print("CPU:")
    benchmark_model(model, test_dataloader, device=torch.device("cpu"))

    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
