# Cityscapes Semantic Segmentation with U-Net/DeepLabV3+ and Quantization

This repository contains code for training and evaluating semantic segmentation models (U-Net, DeepLabV3+) on the Cityscapes dataset. It includes features for model training, post-training static quantization, performance benchmarking, and execution on both local machines and SLURM-based HPC clusters using Apptainer/Singularity containers.

## Author

Marijn de Lange, Technical University Eindhoven, Eindhoven, The Netherlands.
Email: `m.p.d.lange@student.tue.nl`
Codalab username: `Marijndl`

## Features

*   **Semantic Segmentation:** Train models for Cityscapes dataset.
    *   Supports U-Net (`unet.py`)
    *   Supports DeepLabV3+ with various backbones (e.g., ResNet, MobileNetV2, EfficientNet) via `segmentation_models_pytorch` (`train.py`).
*   **Model Optimization:**
    *   **Post-Training Static Quantization:** Reduce model size and potentially speed up inference using PyTorch's quantization tools (`static_quantization.py`).
    *   **Module Fusion:** Fuse compatible layers (Conv-BN-ReLU) for potential speedup.
*   **Experiment Tracking:** Integrated with Weights & Biases (`wandb`) for logging metrics, configurations, and visualizations (`train.py`).
*   **Benchmarking:** Evaluate models based on:
    *   Accuracy (Dice Score)
    *   Inference Speed (Images/Second on CPU/GPU)
    *   Computational Complexity (GMACs)
    *   Model Size (MB)
    *   (Implemented in `utils.py` and used in `train.py`, `static_quantization.py`)
*   **Data Augmentation:** Includes standard augmentations (Resize, Flip) and a custom Motion Blur augmentation (`utils.py`, `visualize_augmentation.py`).
*   **SLURM Cluster Support:** Scripts provided for easy execution on HPC clusters using Apptainer/Singularity containers (`jobscript_slurm.sh`, `download_docker_and_data.sh`, `main.sh`, `quant.sh`).
*   **Containerized Environment:** `Dockerfile` provided to build a consistent environment with necessary dependencies.

## Prerequisites

*   **Software:**
    *   Git
    *   Python 3.8+
    *   A virtual environment manager (like `venv` or `conda`) is highly recommended.
    *   (For SLURM) Apptainer or Singularity installed on the cluster.
*   **Hardware:**
    *   A CUDA-enabled GPU is strongly recommended for training and faster inference/benchmarking.
*   **Accounts:**
    *   GitHub (for cloning the repository)
    *   Weights & Biases (W&B) (for experiment logging - required for training script)

## Installation

Follow the instructions based on your execution environment (Local Machine or SLURM Cluster).

### 1. Clone the Repository

```bash
git https://github.com/Marijndl/NNCV_personal/
cd NNCV_personal/Final\ assignment/
```

*(Refer to `README-Installation.md` for detailed instructions on cloning and potentially setting up your own fork on GitHub if needed).*

### 2. Local Installation

This setup is for running the code directly on your local machine.

1.  **Create and Activate Virtual Environment:**
    ```bash
    # Using venv
    python3 -m venv venv
    source venv/bin/activate
    # On Windows: .\venv\Scripts\activate

    # OR using conda
    # conda create -n cityscapes_env python=3.9
    # conda activate cityscapes_env
    ```

2.  **Install PyTorch:** Install PyTorch separately first, matching your system's CUDA version (if applicable). Follow instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    # Install remaining requirements
    pip install -r ../requirements.txt
    ```
    *Note: `requirements.txt` includes specific versions. If you encounter issues, check compatibility, especially between PyTorch, CUDA, `torchvision`, and `torch_tensorrt`.*

### 3. SLURM Cluster Environment Setup

This setup utilizes an Apptainer/Singularity container, meaning most dependencies are pre-installed within the container.

1.  **Download Container and Data:**
    *   Ensure you are in the `Final assignment` directory on the SLURM cluster.
    *   Make the download script executable (only needs to be done once):
        ```bash
        chmod +x download_docker_and_data.sh
        ```
    *   Submit the download job:
        ```bash
        sbatch download_docker_and_data.sh
        ```
    *   Wait for the job to complete. This will download the dataset into a `data/` subdirectory and the container image as `container.sif`.

2.  **Configure Environment Variables:**
    *   Create or edit the `.env` file in the `Final assignment` directory. This file stores sensitive information like your W&B API key.
    *   Add your Weights & Biases API key:
        ```dotenv
        # .env file content
        WANDB_API_KEY=<your_wandb_api_key>
        # Optional: Specify a directory for W&B logs if needed
        # WANDB_DIR=/path/to/your/wandb/logs
        ```
    *   Replace `<your_wandb_api_key>` with your actual key from your W&B settings.

## Running the Code

The primary method for running experiments is via the SLURM cluster using the provided job scripts. Local execution is also possible.

### Running on SLURM Cluster

The main job submission script is `jobscript_slurm.sh`. It executes other shell scripts (`main.sh` or `quant.sh`) inside the Apptainer container.

1.  **Configure SLURM Job Resources (Optional):**
    *   Edit `jobscript_slurm.sh` to adjust resources like `--time`, `--cpus-per-task`, `--gpus`, or `--partition` based on your needs and cluster availability. Refer to `README-Slurm.md` for parameter explanations.

2.  **Configure Experiment Script (`main.sh` or `quant.sh`):**
    *   **For Training:** Edit `main.sh`.
        *   Adjust hyperparameters passed to `train.py`, such as `--batch-size`, `--epochs`, `--lr`, `--model`, `--decoder`, `--experiment-id`.
        *   Ensure `--data-dir` points to `./data/cityscapes`.
        *   The script first sets up a virtual environment *inside* the container's execution context and installs some dependencies like `fvcore`, `torch_tensorrt` etc. if not present.
    *   **For Quantization:** Edit `quant.sh`.
        *   Ensure `--model-file` points to the correct path of your trained floating-point model checkpoint (e.g., `./checkpoints/<experiment-id>/best_model-....pth`).
        *   Adjust `--batch-size`, `--num-workers` if needed.
        *   Ensure `--data-dir` points to `./data/cityscapes`.
        *   This script also sets up a virtual environment similar to `main.sh`.

3.  **Submit the Job:**
    *   Make the job script executable (only needs to be done once):
        ```bash
        chmod +x jobscript_slurm.sh
        ```
    *   Submit the job to SLURM (ensure `jobscript_slurm.sh` calls the correct script - `main.sh` for training, `quant.sh` for quantization):
        ```bash
        sbatch jobscript_slurm.sh
        ```

4.  **Monitor and Check Results:**
    *   Use `squeue -u <your_username>` to monitor job status.
    *   Check the output log file `slurm-<job_id>.out` for progress and errors.
    *   Check your Weights & Biases project page for live training metrics.
    *   Model checkpoints will be saved in the `checkpoints/<experiment-id>/` directory (for training) or the directory specified by `--model-file`'s parent (for quantization output files like `unet_quantization_scripted_quantized.pth`).

### Running Locally

1.  **Activate your local virtual environment:**
    ```bash
    source venv/bin/activate # or conda activate cityscapes_env
    ```
2.  **Run Training:**
    ```bash
    python train.py \
        --data-dir /path/to/your/local/cityscapes \
        --batch-size 32 \
        --epochs 50 \
        --lr 0.001 \
        --num-workers 4 \
        --seed 42 \
        --experiment-id "local-unet-training" \
        --model "unet" # or "deeplab"
        # Add other arguments as needed, such as --motion-blur
    ```
    *Remember to `wandb login` first if you haven't.*

3.  **Run Quantization:**
    ```bash
    python static_quantization.py \
        --data-dir /path/to/your/local/cityscapes \
        --model-file /path/to/your/trained/float_model.pth \
        --batch-size 16 \
        --num-workers 4 \
        --seed 14052004
    ```
    *Ensure the `--model-file` path is correct.*

## Scripts Overview

*   `train.py`: Main script for training segmentation models (U-Net, DeepLabV3+). Handles data loading, augmentation, training loop, validation, W&B logging, model saving, and final benchmarking.
*   `static_quantization.py`: Performs post-training static quantization on a pre-trained U-Net model. Includes fusion, calibration, conversion, evaluation, and benchmarking.
*   `unet.py`: Defines the U-Net model architecture, including quantization stubs and fusion logic.
*   `utils.py`: Contains utility functions for Dice score calculation, model benchmarking (speed, size, MACs), evaluation loop, data loading helpers, class/color mappings, model loading, and custom data augmentation (`MotionBlurTransform`).
*   `main.sh`: Bash script executed by SLURM (via `jobscript_slurm.sh`) to run the training process (`train.py`) inside the container. Sets up a temporary venv.
*   `quant.sh`: Bash script executed by SLURM (via `jobscript_slurm.sh`) to run the quantization process (`static_quantization.py`) inside the container. Sets up a temporary venv.
*   `jobscript_slurm.sh`: SLURM submission script. Configures resources and executes either `main.sh` or `quant.sh` within the Apptainer container.
*   `download_docker_and_data.sh`: SLURM script to download the Apptainer container (`.sif`) and the Cityscapes dataset from Hugging Face.
*   `Dockerfile`: Defines the build steps for the container image, including OS, system packages, and Python libraries.
*   `requirements.txt`: Lists Python package dependencies.
*   `README-Installation.md`: Detailed setup guide for tools like VSCode, Git, W&B, MobaXTerm.
*   `README-Slurm.md`: Detailed guide for running jobs on the SLURM cluster.
*   `plot_results.py`: Example script to plot comparison metrics (uses hardcoded data from experiments).
*   `visualize_augmentation.py`: Script to visualize the effect of the custom `MotionBlurTransform`.
*   `archive/`: Contains older or alternative experimental scripts (e.g., for TensorRT quantization, pruning, different training setups).

## Configuration

*   `.env`: Stores sensitive keys (like `WANDB_API_KEY`). Create this file in the `Final assignment` directory before running jobs on SLURM.
*   `jobscript_slurm.sh`: Configure SLURM resource allocation (time, nodes, CPUs, GPUs, partition).
*   `main.sh` / `quant.sh`: Configure hyperparameters and script arguments for training/quantization runs.
