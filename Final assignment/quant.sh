wandb login

python3 -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

python3 static_quantization.py \
    --data-dir ./data/cityscapes \
    --model-file "./quant_models/unet_noaug_float.pth" \
    --batch-size 16 \
    --num-workers 10 \
    --seed 14052004 \