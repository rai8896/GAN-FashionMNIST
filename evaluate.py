"""
evaluate.py
===========
Load a trained generator and evaluate:
  1. Visual grid of generated images
  2. Diversity score (std of pixel values across samples)
  3. Upload generator to Hugging Face Hub
"""

import os
import argparse
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import VanillaGenerator, DCGANGenerator


# ─────────────────────────────────────────────
#  Load generator from checkpoint
# ─────────────────────────────────────────────

def load_generator(arch: str, checkpoint_path: str,
                   latent_dim: int = 100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch.lower() == "vanilla":
        G = VanillaGenerator(latent_dim=latent_dim)
    else:
        G = DCGANGenerator(latent_dim=latent_dim)

    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.to(device).eval()
    print(f"[Evaluate] Loaded {arch} generator from {checkpoint_path}")
    return G, device


# ─────────────────────────────────────────────
#  Generate image grid
# ─────────────────────────────────────────────

def generate_grid(G, latent_dim: int, n_images: int = 64,
                  device=None, save_path: str = "generated_grid.png"):
    with torch.no_grad():
        z    = torch.randn(n_images, latent_dim, device=device)
        imgs = G(z)
        if imgs.dim() == 2:                    # Vanilla → reshape
            imgs = imgs.view(-1, 1, 28, 28)

    grid = vutils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np, cmap="gray")
    plt.axis("off")
    plt.title(f"Generated Samples  ({n_images} images)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Grid saved → {save_path}")
    return grid_np


# ─────────────────────────────────────────────
#  Diversity score
# ─────────────────────────────────────────────

def diversity_score(G, latent_dim: int, n_samples: int = 1000, device=None):
    """
    A simple diversity metric: mean pairwise pixel std across samples.
    Higher is better (more diverse outputs).
    """
    with torch.no_grad():
        z    = torch.randn(n_samples, latent_dim, device=device)
        imgs = G(z)
        if imgs.dim() == 2:
            imgs = imgs.view(n_samples, -1)
        else:
            imgs = imgs.view(n_samples, -1)

    imgs_np = imgs.cpu().numpy()
    std_per_pixel = imgs_np.std(axis=0)        # std across samples for each pixel
    diversity     = float(std_per_pixel.mean())
    print(f"[Evaluate] Diversity Score (mean pixel std): {diversity:.4f}")
    return diversity


# ─────────────────────────────────────────────
#  Loss curve plots (from W&B CSV export)
# ─────────────────────────────────────────────

def plot_loss_curves(csv_path: str, save_path: str = "loss_curves.png"):
    """
    If you export W&B run data as CSV, this plots G/D loss curves.
    CSV must have columns: epoch, g_loss, d_loss
    """
    import csv

    epochs, g_losses, d_losses = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            g_losses.append(float(row["g_loss"]))
            d_losses.append(float(row["d_loss"]))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, g_losses, label="Generator Loss",     color="#e74c3c", lw=2)
    plt.plot(epochs, d_losses, label="Discriminator Loss", color="#2980b9", lw=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Loss curves saved → {save_path}")


# ─────────────────────────────────────────────
#  Hugging Face upload
# ─────────────────────────────────────────────

def upload_to_huggingface(
    checkpoint_path: str,
    repo_id: str,
    arch: str,
    config: dict,
    grid_path: str = None,
):
    """
    Uploads generator checkpoint + model card to Hugging Face Hub.

    Args:
        repo_id    : "your-username/your-repo-name"
        arch       : "vanilla" or "dcgan"
        config     : dict of hyperparameters for the model card
        grid_path  : path to sample image grid (optional)
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[HF] Install huggingface_hub: pip install huggingface_hub")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(repo_id, exist_ok=True, repo_type="model")

    # Upload checkpoint
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=os.path.basename(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[HF] Checkpoint uploaded → {repo_id}")

    # Upload grid image (if provided)
    if grid_path and os.path.exists(grid_path):
        api.upload_file(
            path_or_fileobj=grid_path,
            path_in_repo=os.path.basename(grid_path),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[HF] Sample grid uploaded → {repo_id}")

    # Create model card
    card_content = f"""---
tags:
  - pytorch
  - generative
  - image-generation
  - fashion-mnist
  - gan
license: mit
---

# {arch.upper()} GAN — Fashion-MNIST

Trained on Fashion-MNIST to generate grayscale clothing images (28×28).

## Config
```json
{config}
```

## Architecture
- **Generator**: {'Fully-connected layers' if arch == 'vanilla' else 'Transposed convolutions + BatchNorm + ReLU'}
- **Discriminator**: {'Fully-connected layers' if arch == 'vanilla' else 'Conv layers + LeakyReLU'}

## Sample Outputs
![Generated Samples](generated_grid.png)

## Usage
```python
import torch
from models import {'VanillaGenerator' if arch == 'vanilla' else 'DCGANGenerator'}

G = {'VanillaGenerator' if arch == 'vanilla' else 'DCGANGenerator'}(latent_dim=100)
G.load_state_dict(torch.load("generator.pth", map_location="cpu"))
G.eval()

z = torch.randn(16, 100)
with torch.no_grad():
    imgs = G(z)  # shape: (16, 784) or (16, 1, 28, 28)
```
"""
    card_path = "README.md"
    with open(card_path, "w") as f:
        f.write(card_content)

    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[HF] Model card uploaded → https://huggingface.co/{repo_id}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained GAN generator")
    p.add_argument("--arch",        type=str, required=True, choices=["vanilla", "dcgan"])
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--latent_dim",  type=int, default=100)
    p.add_argument("--n_images",    type=int, default=64)
    p.add_argument("--output_dir",  type=str, default="./eval_outputs")
    p.add_argument("--hf_repo",     type=str, default=None,
                   help="HuggingFace repo ID  e.g.  username/my-gan")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    G, device = load_generator(args.arch, args.checkpoint, args.latent_dim)

    grid_path = os.path.join(args.output_dir, "generated_grid.png")
    generate_grid(G, args.latent_dim, args.n_images, device, grid_path)

    score = diversity_score(G, args.latent_dim, n_samples=1000, device=device)

    if args.hf_repo:
        config_dict = {"arch": args.arch, "latent_dim": args.latent_dim}
        upload_to_huggingface(
            checkpoint_path = args.checkpoint,
            repo_id         = args.hf_repo,
            arch            = args.arch,
            config          = config_dict,
            grid_path       = grid_path,
        )
