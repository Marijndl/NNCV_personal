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

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training" \