Dynamic VAE
<p align="center">
  <img src="path/to/reconstructions.png" alt="Reconstructions Grid" width="600"/>
</p>

A Variational Autoencoder (VAE) incorporating residual connections and self-attention mechanisms for enhanced image generation and reconstruction.

## ✨ Features
* **Dynamic Architecture:** Easily configure encoder/decoder channel depths.
* **Residual Blocks:** Improves training stability and feature propagation in deep networks.
* **Self-Attention:** Captures long-range dependencies for improved image coherence.
* **Perceptual Loss:** Leverages VGG features for visually superior reconstructions.
* **WandB Integration:** Seamless experiment tracking and visualization.
* **Multi-GPU Support:** Utilizes `torch.nn.DataParallel` for accelerated training.

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch

### Installation
```bash
git clone https://github.com/yourusername/your-vae-repo.git
cd your-vae-repo
pip install -r requirements.txt
```
