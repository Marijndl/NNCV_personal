wandb login

# Create a virtual environment in home directory (if not already created)
export VENV_DIR="$HOME/venv_quantization"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install fvcore kornia segmentation_models_pytorch thop torch_tensorrt
else
    source $VENV_DIR/bin/activate
fi

python3 -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

python3 static_quantization.py \
    --data-dir ./data/cityscapes \
    --model-file "./quant_models/unet_noaug_float.pth" \
    --batch-size 16 \
    --num-workers 10 \
    --seed 14052004 \