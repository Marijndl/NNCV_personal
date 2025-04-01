import torch
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
import matplotlib.pyplot as plt
from utils import MotionBlurTransform
from torch.utils.data import DataLoader


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return torch.tensor([id_to_trainid.get(int(x), 255) for x in label_img.flatten()], dtype=torch.uint8).reshape(label_img.shape)

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

transform_load = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),
])

dataset = Cityscapes("D:/Cityscapes", split="train", mode="fine", target_type="semantic", transforms=transform_load)
train_dataset = wrap_dataset_for_transforms_v2(dataset)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

num_samples = 2
fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))

iterator = iter(train_dataloader)

for i in range(num_samples):
    image, mask = next(iterator)
    image = image.squeeze(0)  # Remove batch dimension
    mask = mask.squeeze(0)

    motion_blur_transform = MotionBlurTransform()
    trans_image, trans_mask = motion_blur_transform((image, mask))

    # Convert mask to train IDs first
    mask_train = convert_to_train_id(mask.clone())
    trans_mask_train = convert_to_train_id(trans_mask.clone())

    # Adjust shape for convert_train_id_to_color
    mask_train_4d = mask_train.unsqueeze(0) # (1, 1, H, W)
    trans_mask_train_4d = trans_mask_train.unsqueeze(0)

    orig_mask_colored = convert_train_id_to_color(mask_train_4d)
    trans_mask_colored = convert_train_id_to_color(trans_mask_train_4d)

    axes[i, 0].imshow(image.permute(1, 2, 0).detach().cpu().numpy())
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(trans_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    axes[i, 1].set_title("Transformed Image (Motion Blur)")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(orig_mask_colored.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    axes[i, 2].set_title("Original Mask")
    axes[i, 2].axis('off')

    axes[i, 3].imshow(trans_mask_colored.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    axes[i, 3].set_title("Transformed Mask")
    axes[i, 3].axis('off')

plt.tight_layout()
plt.show()
