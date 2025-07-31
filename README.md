# Dynamic & Enhanced Variational Autoencoder (VAE)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Dynamic & Enhanced Variational Autoencoder (VAE)](#dynamic--enhanced-variational-autoencoder-vae)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Train the Standard VAE](#train-the-standard-vae)
    - [Train the Enhanced VAE](#train-the-enhanced-vae)
  - [WandB Integration](#wandb-integration)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

This repository presents a **Dynamic and Enhanced Variational Autoencoder (VAE)** implemented in PyTorch. The VAE is designed with a highly configurable convolutional architecture that automatically adjusts to various input image sizes and channel configurations (grayscale or color). Furthermore, it features an "enhanced" mode that incorporates modern deep learning techniques like Residual Blocks, Self-Attention Mechanisms, and a Perceptual Loss component (using a pre-trained VGG network) for generating higher-quality and perceptually richer reconstructions.

The project emphasizes modularity, ease of use, and robust experiment tracking through Weights & Biases (WandB).

## Features

* **Dynamic Architecture:** The VAE encoder and decoder are built dynamically based on user-defined channel lists and input image dimensions (`input_height`, `input_width`), adapting to various image sizes (e.g., 28x28, 64x64, 128x128, or even non-square images).
* **Enhanced VAE Capabilities:**
    * **Residual Blocks:** Improves training stability and allows for deeper networks.
    * **Self-Attention Modules:** Captures long-range dependencies within feature maps, leading to more coherent generated outputs.
    * **Perceptual Loss:** Utilizes a pre-trained VGG-16 network to measure the perceptual similarity between real and reconstructed images, encouraging more visually appealing results beyond pixel-wise metrics.
* **Universal Image Dataset:** A flexible `UniversalImageDataset` class that automatically discovers common image formats (`.png`, `.jpg`, `.jpeg`, `.bmp`), handles `grayscale` or `color` conversion, and resizes images to a `target_size`.
* **Robust Training Pipeline:**
    * **Weights & Biases (WandB) Integration:** Seamless logging of training/validation metrics, model graphs, gradients, and sample image reconstructions for comprehensive experiment tracking.
    * **Learning Rate Scheduling:** `ReduceLROnPlateau` for adaptive learning rate adjustments based on validation loss.
    * **Gradient Clipping:** Helps stabilize training and prevent exploding gradients.
    * **Checkpointing:** Automatically saves the latest and best-performing models based on validation loss.
    * **Multi-GPU Support:** Supports `torch.nn.DataParallel` for efficient training on multiple GPUs.
* **Modular Design:** Separated components for dataset handling, VAE architecture (standard vs. enhanced), and the training loop for clear organization and extensibility.

## Project Structure

```

.
├── configs/
│   └── vae\_config.json        \# Configuration file for training parameters
├── dataset.py                 \# Universal image dataset class
├── vae.py                     \# Standard Dynamic VAE model definition
├── enhanced\_vae.py            \# Enhanced Dynamic VAE model with Residual, Attention, Perceptual Loss
└── run\_vae.py                 \# Main script for training and validation

````

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mehdinejjar86/dynamicvae](https://github.com/mehdinejjar86/dynamicvae)
    cd your-vae-project
    ```
    (Remember to replace `yourusername/your-vae-project` with your actual repository path)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121) # For CUDA 12.1, adjust for your CUDA/CPU setup
    pip install wandb Pillow tqdm
    ```
    *Note: For PyTorch installation, refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the exact command that matches your operating system and CUDA version (or CPU only). Also MPS is supported.*

## Data Preparation

Place your image dataset into a directory. By default, the `run_vae.py` script expects data in `./train` as specified in `vae_config.json`.

Example:
````

your-vae-project/
├── train/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
└── ...

````

## Configuration

All training parameters are managed in the `configs/vae_config.json` file. Here's an example of the default configuration:

```json
{
  "input_height": 28,
  "input_width": 28,
  "encoder_channels": [32, 64, 128],
  "decoder_channels": [128, 64, 32],
  "latent_dim": 20,
  "batch_size": 128,
  "lr": 1e-3,
  "weight_decay": 1e-5,
  "grad_clip": 1.0,
  "epochs": 100,
  "data_path": "./train",
  "image_mode": "grayscale",
  "log_interval": 10,
  "image_log_interval": 15,
  "project_name": "VAE-Training",
  "tags": ["grayscale", "beta-vae"],
  "save_dir": "./vae_checkpoints",
  "use_dataparallel": true,
  "num_workers": 4,
  "perceptual_weight": 0.1,
  "kld_weight": 0.5
}
````

**Parameters Explained:**

  * `input_height`, `input_width`: Target dimensions for resizing input images.
  * `encoder_channels`: List of output channels for each convolutional layer in the encoder. The first element of `decoder_channels` should match the last element of `encoder_channels` for the latent space mapping.
  * `decoder_channels`: List of output channels for each convolutional transpose layer in the decoder.
  * `latent_dim`: Dimensionality of the latent space (z-vector).
  * `batch_size`: Number of samples per batch.
  * `lr`: Initial learning rate for the Adam optimizer.
  * `weight_decay`: L2 regularization coefficient.
  * `grad_clip`: Maximum norm for gradient clipping.
  * `epochs`: Number of training epochs.
  * `data_path`: Path to your dataset directory.
  * `image_mode`: `grayscale` or `color` for image loading.
  * `log_interval`: How often (in batches) to log metrics to WandB during training.
  * `image_log_interval`: How often (in epochs) to log reconstruction examples to WandB.
  * `project_name`: Name of the project in WandB.
  * `tags`: Tags for the WandB run.
  * `save_dir`: Directory to save model checkpoints and metrics.
  * `use_dataparallel`: Set to `true` to enable `nn.DataParallel` if multiple GPUs are available.
  * `num_workers`: Number of subprocesses to use for data loading.
  * `perceptual_weight`: Weight for the perceptual loss component (only for Enhanced VAE).
  * `kld_weight`: Weight for the KL Divergence loss component.

## Usage

To train the VAE model, run `run_vae.py` from your terminal.

### Train the Standard VAE

To train the standard VAE model (using `vae.py`), simply run:

```bash
python run_vae.py --config configs/vae_config.json
```

### Train the Enhanced VAE

To leverage the enhanced VAE features (Residual Blocks, Attention, Perceptual Loss from `enhanced_vae.py`), add the `--enhanced` flag:

```bash
python run_vae.py --config configs/vae_config.json --enhanced
```

## WandB Integration

This project is fully integrated with Weights & Biases (WandB) for comprehensive experiment tracking. During training, WandB will log:

  * Training and validation loss metrics (total loss, reconstruction loss, KLD loss).
  * Learning rate.
  * Gradient flow.
  * Model architecture (graph).
  * Sample input images and their reconstructions.
  * Dataset statistics (initial logging in `dataset.py`).

Ensure you have a WandB account and are logged in (`wandb login`) before running the training script. Your runs will appear under the `project_name` specified in your `vae_config.json`.

## Results

<p align="center">
  <img src="image/grid.png" alt="Reconstructions Grid" width="600"/>
</p>

## Contributing

Contributions are welcome\! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](#) file for details.
