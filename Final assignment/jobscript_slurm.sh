#!/bin/bash
#SBTACH --nodes=1
#SBTACH --ntasks=1
#SBTACH --cpus-per-task=18
#SBTACH --gpus=1
#SBTACH --partition=gpu
#SBTACH --time=04:20:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh