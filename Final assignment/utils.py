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
        pred_class = (preds == class_id).float()  # Binary mask for class
        target_class = (targets == class_id).float()  # Binary mask for class

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)

    mean_dice = torch.mean(torch.stack(dice_per_class))  # Mean Dice over classes
    return mean_dice.item()