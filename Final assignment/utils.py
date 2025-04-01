import torch
import os
from unet import UNet
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from tqdm import tqdm
import torch
import torchvision.transforms.v2 as T
import kornia.filters as kf

class MotionBlurTransform(T.Transform):
    def __call__(self, sample):
        image, mask = sample
        p = torch.rand(1).item()
        if p > 0.5:
            # Sample c between 1 and C, C is 19 for Cityscapes (0-18, ignoring 255)
            C = 19
            c = torch.randint(1, C + 1, (1,)).item()
            # Get unique classes, filter out 255
            classes = torch.unique(mask[mask != 255])
            if len(classes) < c:
                c = len(classes)  # Adjust if not enough classes
            if c > 0:
                # Randomly select c classes
                selected_classes = classes[torch.randperm(len(classes))[:c]]
                print(f"Selected classes: {selected_classes}")

                # Create Mf by summing masks for selected classes
                Mf = torch.zeros_like(mask, dtype=torch.float32)
                for cls in selected_classes:
                    Mf += (mask == cls).float()
                Mf[Mf > 0] = 1  # Binary mask where selected classes are

                # Generate motion blur with random parameters
                kernel_size = torch.randint(5, 9, (1,)).item() * 2 + 1  #only odd numbers
                angle = torch.rand(1).item() * 360  # 0-360 degrees
                direction = torch.rand(1).item() * 2 - 1  # -1 to 1
                # Apply motion blur to the part where Mf=1
                blurred_part = image * Mf.unsqueeze(0)  # [C, H, W]
                blurred_part = kf.motion_blur(blurred_part, kernel_size, angle, direction, border_type='reflect')
                # Non-blurred part where Mf=0
                non_blurred_part = image * (1 - Mf.unsqueeze(0))
                # Combine: where Mf=1 use blurred, else original
                image = blurred_part + non_blurred_part
        return image, mask

def dice_score(preds, targets, num_classes=19, ignore_index=255, smooth=1e-6):
    """
    Computes the mean DICE score across all classes, ignoring the specified index.

    Args:
        preds (torch.Tensor): Model predictions (logits or probabilities) with shape (B, C, H, W).
        targets (torch.Tensor): Ground truth segmentation masks with shape (B, H, W).
        num_classes (int): Number of segmentation classes.
        ignore_index (int): Index to ignore in ground truth.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        mean_dice (float): The mean DICE score across all classes.
    """
    preds = torch.argmax(preds, dim=1)  # Convert logits/probs to class labels (B, H, W)
    valid_mask = targets != ignore_index  # Mask out ignore index in targets

    dice_per_class = []
    for class_id in range(num_classes):
        pred_class = (preds == class_id) & valid_mask  # Binary mask for predicted class, ignoring index
        target_class = (targets == class_id) & valid_mask  # Binary mask for target class, ignoring index

        pred_class = pred_class.float()
        target_class = target_class.float()

        intersection = (pred_class * target_class).sum(dim=(1, 2))  # Sum over spatial dims per batch
        union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))

        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Avoid including empty classes in the mean
        valid_class_mask = union > 0  # Avoid empty class issue
        if valid_class_mask.any():
            dice_per_class.append(dice[valid_class_mask])

    if dice_per_class:
        mean_dice = torch.cat(dice_per_class).mean().item()  # Mean Dice over valid classes
    else:
        mean_dice = 0.0  # Return 0 if no classes were valid

    return mean_dice

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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


def evaluate(model, criterion, data_loader, neval_batches, device='cpu', num_classes=19):
    model.eval()
    dice_meter = AverageMeter('Dice', ':6.6f')  # Track Dice scores
    cnt = 0

    with torch.no_grad():
        for i, (image, target) in tqdm(enumerate(data_loader), total=neval_batches-1, desc="Evaluating", leave=True):
            target = convert_to_train_id(target)  # Convert class IDs to train IDs
            image, target = image.to(device), target.to(device)

            target = target.long().squeeze(1)  # Remove channel dimension
            output = model(image)
            loss = criterion(output, target)
            cnt += 1

            dice = dice_score(output, target, num_classes)
            dice_meter.update(dice, image.size(0))  # Update average meter

            if cnt >= neval_batches:
                break

    return dice_meter.avg  # Return the final average Dice score

def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("maxpool_conv.1", "double_conv")\
                     .replace(".conv.double_conv", ".double_conv.double_conv")
                     # .replace( "down3", "down3.double_conv")\
                     # .replace( "down4", "down4.double_conv")\
                     # .replace("up1", "up1.double_conv")\
                     # .replace("up2", "up2.double_conv")\
                     # .replace("up3", "up3.double_conv")\
                     # .replace("up4", "up4.double_conv")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def load_model(model_file, quantize=False):
    model = UNet(in_channels=3, n_classes=19, quantize=quantize)
    # Load the checkpoint
    state_dict = torch.load(model_file, map_location='cpu')  # Load to CPU initially

    # Rename state dict keys
    state_dict = rename_state_dict_keys(state_dict)

    # Load the modified state dict into the model
    model.load_state_dict(state_dict, strict=True)  # set strict=False to ignore size mismatch
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')