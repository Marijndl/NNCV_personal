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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomHorizontalFlip,
    RandomResizedCrop
)
from utils import * 

from unet import UNet


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
    parser.add_argument("--T", type=int, default=2, help="Temperature for smoothing")
    parser.add_argument("--st-loss", type=int, default=0.25/50000, help="Relative weight of soft target loss (teacher loss).")
    parser.add_argument("--ce-loss", type=int, default=1, help="Relative weight of cross entropy loss (training loss).")

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

    # Define common transforms (resize to 1024 first, then resize for student)
    transform_common = Compose([
        ToImage(),
        RandomResizedCrop(size=(768, 768), scale=(0.4, 0.8), antialias=True),
        RandomHorizontalFlip(p=0.5),
        
        # Resize((768, 768), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
    ])

    # Teacher-specific normalization
    transform_teacher = Compose([
        ToImage(),
        # Resize((768, 768), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
    ])

    # Student-specific normalization and downscaling
    transform_student = Compose([
        Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
        Normalize((0.2869, 0.3251, 0.2839), (0.1869, 0.1901, 0.1872)),
    ])

    transform_label = Compose([
        ToImage(),
        Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.uint8, scale=True),
    ])

    class DistillationDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __getitem__(self, idx):
            image, label = self.dataset[idx]

            # Add data augmentation
            image = transform_common(image)

            # Create teacher input (1024x1024 normalized)
            teacher_input = transform_teacher(image.clone())
            
            # Apply common transforms
            label = transform_label(label)

            # Create student input (384x384 normalized)
            student_input = transform_student(image.clone())

            return student_input, teacher_input, label

        def __len__(self):
            return len(self.dataset)

    
    # Load raw dataset
    raw_train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine", target_type="semantic"
    )

    raw_valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine", target_type="semantic"
    )

    # Ensure compatibility with transform handling
    train_dataset = DistillationDataset(wrap_dataset_for_transforms_v2(raw_train_dataset))
    valid_dataset = DistillationDataset(wrap_dataset_for_transforms_v2(raw_valid_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Define the model
    model = UNet(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Teacher model
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")
    teacher_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768").to(device)
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # Define warmup function
    def lr_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10  # Linear warmup
        else:
            return 1  # After warmup, let CosineAnnealing take over
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 10, eta_min=1e-6)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images_student, images_teacher, labels) in enumerate(train_dataloader):
            
            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images_student, images_teacher, labels = images_student.to(device), images_teacher.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()

            with torch.no_grad():
                inputs = feature_extractor(images=images_teacher, return_tensors="pt").to(device)
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            student_logits = model(images_student)

            soft_targets = nn.functional.softmax(teacher_logits / args.T, dim=1)
            soft_prob = nn.functional.log_softmax(student_logits / args.T, dim=1)

            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (args.T**2)
            label_loss = criterion(student_logits, labels)

            loss = args.st_loss * soft_targets_loss + args.ce_loss * label_loss
            
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # Step the scheduler
        if epoch < 10:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            dice_scores = []
            for i, (images, _, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                dice = dice_score(outputs, labels)
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
            
            valid_loss = sum(losses) / len(losses)
            valid_dice = sum(dice_scores) / len(dice_scores)
            wandb.log({
                "valid_loss": valid_loss,
                "valid_dice_score": valid_dice,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
