from argparse import ArgumentParser

import torch.nn as nn

from utils import *


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--model-file", type=str, default="./quant_models/unet_noaug_float.pth",
                        help="Path to the float model")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


def main(args):
    # Set seed for reproducability
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    saved_model_dir = "/".join(args.model_file.split("/")[:-1]) + "/"
    float_model_file = args.model_file
    scripted_float_model_file = 'unet_quantization_scripted.pth'
    scripted_quantized_model_file = 'unet_quantization_scripted_quantized.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_batch_size = args.batch_size
    # Load the dataset and make a split for training and validation
    # Define the transforms to apply to the data
    transform = Compose([
        ToImage(),
        Resize((512, 512), antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=transform,
    )
    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transforms=transform,
    )
    test_dataset = Cityscapes(
        args.data_dir,
        split="test",
        mode="fine",
        target_type="semantic",
        transforms=transform,
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)
    test_dataset = wrap_dataset_for_transforms_v2(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    float_model = load_model(float_model_file, quantize=False).to("cpu")

    # Evaluate the model on artificial test set.
    float_model.to('cpu')
    float_model.eval()

    num_eval_batches = 20
    dice_avg = evaluate(float_model, criterion, test_dataloader, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on test dataset, {num_eval_batches * args.batch_size} images, dice: {dice_avg}')

    print("CPU:")
    float_model.to('cpu')
    benchmark_model(float_model, test_dataloader, device=torch.device("cpu"))

    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    print('\n Inc Block: Before fusion \n\n', float_model.inc.double_conv)
    float_model.eval()

    print("--------------------------------------")

    # Fuses modules
    float_model.to('cpu')
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inc Block: After fusion\n\n', float_model.inc.double_conv)

    print("Size of baseline model")
    print_size_of_model(float_model)

    num_eval_batches = 64

    float_model = float_model.to(device)
    dice_avg = evaluate(float_model, criterion, valid_dataloader, device=device, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches * eval_batch_size} images, dice: {dice_avg}')
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

    # Fine tune model on training data
    per_channel_quantized_model = load_model(float_model_file, quantize=True)
    per_channel_quantized_model = per_channel_quantized_model.to('cpu')
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print(per_channel_quantized_model.qconfig)

    num_calibration_batches = 100

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model, criterion, train_dataloader, num_calibration_batches)
    torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)

    # Evaluation after quantization:
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)

    per_channel_quantized_model.to('cpu')
    per_channel_quantized_model.eval()
    print(f"Size of quantized model")
    print_size_of_model(per_channel_quantized_model)

    num_eval_batches = 20
    dice_avg = evaluate(per_channel_quantized_model, criterion, test_dataloader, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on test dataset, {num_eval_batches * args.batch_size} images, dice: {dice_avg}')

    print("CPU:")
    per_channel_quantized_model.to('cpu')
    benchmark_model(per_channel_quantized_model, test_dataloader, device=torch.device("cpu"))

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
