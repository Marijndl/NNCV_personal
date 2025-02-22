wandb login

python3 train.py \
    --data-dir /home/scur1345/NNCV_personal/Final assignment/data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 4 \
    --seed 42 \
    --experiment-id "unet-training-1" \