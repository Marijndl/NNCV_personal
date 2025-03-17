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

    dice_scores = torch.zeros(num_classes, device=preds.device)
    
    for class_id in range(num_classes):
        pred_class = (preds == class_id).float()
        target_class = (targets == class_id).float()

        intersection = (pred_class * target_class).sum(dim=(1, 2), keepdim=True)
        union = pred_class.sum(dim=(1, 2), keepdim=True) + target_class.sum(dim=(1, 2), keepdim=True)

        dice = (2. * intersection + smooth) / (union + smooth)

        valid_mask = union > 0  # Mask to exclude empty classes
        dice_scores[class_id] = dice.masked_select(valid_mask).mean() if valid_mask.any() else 0.0

    mean_dice = dice_scores[dice_scores > 0].mean().item() if (dice_scores > 0).any() else 0.0
    return mean_dice
