# Experiment 9: GAN vs DCGAN on Fashion-MNIST

Comprehensive PyTorch implementation comparing **Vanilla GAN** and **DCGAN** across
**3 loss functions** × **3 optimizers** = **18 experiment combinations**.

---

## Project Structure

```
gan_experiment/
│
├── models.py           # VanillaGAN + DCGAN architectures
├── losses.py           # BCE, LSGAN, WGAN loss functions
├── dataset.py          # Fashion-MNIST dataloader
├── trainer.py          # Training loop + W&B logging
├── run_experiments.py  # Runs all 18 combinations
├── evaluate.py         # Generate samples + HuggingFace upload
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/gan-fashionmnist
cd gan-fashionmnist

# 2. Install dependencies
pip install -r requirements.txt

# 3. Login to Weights & Biases
wandb login

# 4. Login to Hugging Face
huggingface-cli login
```

---

## Run Experiments

### Run ALL 18 combinations
```bash
python run_experiments.py --epochs 50 --batch_size 64
```

### Run a SINGLE experiment
```bash
python run_experiments.py --arch dcgan --loss bce --optimizer adam --epochs 50
```

### Run only DCGAN experiments
```bash
python run_experiments.py --arch dcgan --epochs 50
```

### Enable data augmentation
```bash
python run_experiments.py --augment --epochs 50
```

---

## Evaluate & Generate Samples

```bash
# Generate image grid + diversity score
python evaluate.py \
  --arch dcgan \
  --checkpoint ./checkpoints/dcgan_bce_adam_best_G.pth \
  --n_images 64 \
  --output_dir ./eval_outputs

# Upload to Hugging Face
python evaluate.py \
  --arch dcgan \
  --checkpoint ./checkpoints/dcgan_bce_adam_best_G.pth \
  --hf_repo your-username/dcgan-fashionmnist
```

---

## Architectures

### Vanilla GAN
| Component     | Architecture |
|---------------|--------------|
| Generator     | Linear(100→256→512→1024→784) + LeakyReLU + Tanh |
| Discriminator | Linear(784→1024→512→256→1) + LeakyReLU + Dropout |

### DCGAN
| Component     | Architecture |
|---------------|--------------|
| Generator     | Linear → ConvTranspose2d ×3 + BatchNorm + ReLU + Tanh |
| Discriminator | Conv2d ×3 + BatchNorm + LeakyReLU → scalar |

---

## Loss Functions

| Loss   | Formula | Advantage |
|--------|---------|-----------|
| BCE    | `-[y·log(D(x)) + (1-y)·log(1-D(G(z)))]` | Simple baseline |
| LSGAN  | `0.5·E[(D(x)-1)²] + 0.5·E[D(G(z))²]` | Avoids vanishing gradients |
| WGAN   | `E[D(G(z))] - E[D(x)]` + weight clipping | Best stability, no mode collapse |

---

## Optimizers

| Optimizer | LR     | Notes |
|-----------|--------|-------|
| Adam      | 2e-4   | β₁=0.5 (standard for GANs) |
| RMSprop   | 2e-4   | Good for non-stationary objectives |
| SGD       | 2e-4   | Momentum=0.9; slowest convergence |

> **For WGAN**: LR is clamped to ≤5e-5 as per the paper.

---

## Experiment Matrix (18 runs)

| Arch    | Loss  | Optimizer | Expected Behavior |
|---------|-------|-----------|-------------------|
| Vanilla | BCE   | Adam      | Baseline; may oscillate |
| Vanilla | LSGAN | Adam      | Smoother than BCE |
| Vanilla | WGAN  | RMSprop   | Most stable Vanilla |
| DCGAN   | BCE   | Adam      | Clear improvement |
| DCGAN   | LSGAN | Adam      | Best visual quality |
| DCGAN   | WGAN  | RMSprop   | Most stable overall |
| ...     | ...   | ...       | ... |

---

## W&B Tracking

Each run logs:
- `g_loss` — Generator loss per epoch
- `d_loss` — Discriminator loss per epoch
- `generated_imgs` — Image grid every epoch

View all runs: [W&B Project Link](https://wandb.ai/<your-username>/GAN-FashionMNIST)

---

## Analysis

### Why DCGAN > Vanilla GAN?
- **Convolutions** respect spatial structure — nearby pixels are processed together
- **BatchNorm** stabilizes gradient flow → faster, more stable training
- **No fully-connected layers** in discriminator → fewer parameters, less overfitting

### Loss Function Comparison
- **BCE**: Simple but suffers vanishing gradients when discriminator is too strong
- **LSGAN**: MSE-based loss penalizes samples far from boundary → smoother gradient
- **WGAN**: Wasserstein distance is continuous and differentiable everywhere → most stable

### Optimizer Comparison
- **Adam**: β₁=0.5 reduces momentum, prevents oscillation — widely recommended for GANs
- **RMSprop**: Good for WGAN (used in original paper); adaptive LR helps with sparse gradients
- **SGD**: Slowest convergence; prone to oscillation in adversarial training

### Training Challenges
| Challenge | Cause | Fix |
|-----------|-------|-----|
| Mode Collapse | G always outputs same image | WGAN loss, diverse mini-batches |
| Vanishing Gradient | D too strong | Label smoothing, LSGAN/WGAN loss |
| Oscillation | G and D competing too aggressively | Lower LR, β₁=0.5 for Adam |

---

## Results Summary

Results saved in `./checkpoints/results_<timestamp>.json` after `run_experiments.py`.

---

## Submission Checklist

- [x] GitHub repository with code + README
- [ ] [Weights & Biases project link](https://wandb.ai/<your-username>/GAN-FashionMNIST)
- [ ] [Hugging Face model link](https://huggingface.co/<your-username>/gan-fashionmnist)
- [ ] Generated sample images
- [ ] Loss curve screenshots

---

## References

1. Goodfellow et al. (2014) — [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
2. Radford et al. (2015) — [DCGAN](https://arxiv.org/abs/1511.06434)
3. Mao et al. (2017)    — [LSGAN](https://arxiv.org/abs/1611.04076)
4. Arjovsky et al. (2017) — [WGAN](https://arxiv.org/abs/1701.07875)
