import torch

def dice_score(preds, targets, num_classes=19, smooth=1e-6):
    """
    Computes the mean DICE score across all classes.

    Args:
        preds (torch.Tensor): Model predictions (logits or probabilities) with shape (B, C, H, W).
        targets (torch.Tensor): Ground truth segmentation masks with shape (B, H, W).
        num_classes (int): Number of segmentation classes.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        mean_dice (float): The mean DICE score across all classes.
    """
    preds = torch.argmax(preds, dim=1)  # Convert logits/probs to class labels (B, H, W)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_class = (preds == class_id).float()  # Binary mask for predicted class
        target_class = (targets == class_id).float()  # Binary mask for target class

        intersection = (pred_class * target_class).sum(dim=(1, 2))  # Sum over spatial dims per batch
        union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))

        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Avoid including empty classes in the mean
        valid_mask = union > 0  # Avoid empty class issue
        if valid_mask.any():
            dice_per_class.append(dice[valid_mask])

    if dice_per_class:
        mean_dice = torch.cat(dice_per_class).mean().item()  # Mean Dice over valid classes
    else:
        mean_dice = 0.0  # Return 0 if no classes were valid

    return mean_dice
