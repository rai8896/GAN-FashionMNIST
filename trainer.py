"""
trainer.py - GAN training loop with Weights & Biases logging
Supports: Vanilla GAN / DCGAN  ×  BCE / LSGAN / WGAN  ×  SGD / RMSprop / Adam
"""

import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import wandb

wandb.login(key="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")

from models  import (VanillaGenerator, VanillaDiscriminator,
                     DCGANGenerator,   DCGANDiscriminator, weights_init)
from losses  import get_loss_fn
from dataset import get_dataloaders


# ─────────────────────────────────────────────
#  Optimizer factory
# ─────────────────────────────────────────────

def get_optimizer(name: str, params, lr: float, wgan: bool = False):
    """
    Returns the requested optimizer.
    For WGAN, learning rate is clamped to a smaller value per the paper.
    """
    if wgan:
        lr = min(lr, 5e-5)          # WGAN paper recommendation

    name = name.upper()
    if name == "ADAM":
        # β1=0.5 is standard for GANs (reduces oscillation)
        return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    elif name == "RMSPROP":
        return torch.optim.RMSprop(params, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose Adam, RMSprop, SGD.")


# ─────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────

def build_models(arch: str, latent_dim: int, device):
    arch = arch.upper()
    if arch == "VANILLA":
        G = VanillaGenerator(latent_dim=latent_dim).to(device)
        D = VanillaDiscriminator().to(device)
    elif arch == "DCGAN":
        G = DCGANGenerator(latent_dim=latent_dim).to(device)
        D = DCGANDiscriminator().to(device)
        G.apply(weights_init)
        D.apply(weights_init)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose Vanilla or DCGAN.")
    return G, D


# ─────────────────────────────────────────────
#  Fixed noise for visualization
# ─────────────────────────────────────────────

def make_fixed_noise(n: int, latent_dim: int, device):
    return torch.randn(n, latent_dim, device=device)


# ─────────────────────────────────────────────
#  Single-epoch training step
# ─────────────────────────────────────────────

def train_one_epoch(G, D, train_loader, loss_fn,
                    opt_G, opt_D, latent_dim, device,
                    n_critic: int = 1):
    """
    Trains G and D for one epoch.

    Args:
        n_critic: How many D steps per G step (WGAN uses 5).
    Returns:
        avg_g_loss, avg_d_loss
    """
    G.train(); D.train()
    total_g, total_d, steps = 0.0, 0.0, 0

    for batch_idx, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # ── Discriminator step ──────────────────
        for _ in range(n_critic):
            z          = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs  = G(z).detach()

            # Reshape for Vanilla GAN discriminator
            if fake_imgs.dim() == 2:           # (B, 784)
                real_in = real_imgs.view(batch_size, -1)
                fake_in = fake_imgs
            else:                              # (B, 1, 28, 28)
                real_in = real_imgs
                fake_in = fake_imgs

            real_logits = D(real_in)
            fake_logits = D(fake_in)

            d_loss = loss_fn.discriminator_loss(real_logits, fake_logits, device)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # WGAN weight clipping
            if loss_fn.clip_weights:
                loss_fn.clip(D)

        # ── Generator step ──────────────────────
        z         = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)

        if fake_imgs.dim() == 2:
            fake_in = fake_imgs
        else:
            fake_in = fake_imgs

        fake_logits = D(fake_in)
        g_loss = loss_fn.generator_loss(fake_logits, device)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        total_g += g_loss.item()
        total_d += d_loss.item()
        steps   += 1

    return total_g / steps, total_d / steps


# ─────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────

def train_experiment(config: dict, project_name: str = "GAN-FashionMNIST"):
    """
    Runs a single GAN experiment defined by `config`.

    config keys:
        arch        : "vanilla" | "dcgan"
        loss        : "bce"    | "lsgan" | "wgan"
        optimizer   : "adam"   | "rmsprop" | "sgd"
        latent_dim  : int  (default 100)
        epochs      : int  (default 50)
        batch_size  : int  (default 64)
        lr          : float (default 2e-4)
        augment     : bool (default False)
        data_dir    : str  (default "./data")
        save_dir    : str  (default "./checkpoints")
        n_critic    : int  (default 1; use 5 for WGAN)
    """

    # ── Defaults ────────────────────────────────
    cfg = {
        "latent_dim" : 100,
        "epochs"     : 50,
        "batch_size" : 64,
        "lr"         : 2e-4,
        "augment"    : False,
        "data_dir"   : "./data",
        "save_dir"   : "./checkpoints",
        "n_critic"   : 1,
    }
    cfg.update(config)

    if cfg["loss"].upper() == "WGAN" and cfg.get("n_critic", 1) == 1:
        cfg["n_critic"] = 5          # WGAN needs more D steps

    run_name = f"{cfg['arch']}_{cfg['loss']}_{cfg['optimizer']}"
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ── W&B init ────────────────────────────────
    run = wandb.init(
        project = project_name,
        name    = run_name,
        config  = cfg,
        reinit  = True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Device     : {device}")
    print(f"{'='*60}")

    # ── Data ────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        data_dir   = cfg["data_dir"],
        batch_size = cfg["batch_size"],
        augment    = cfg["augment"],
    )

    # ── Models ──────────────────────────────────
    G, D = build_models(cfg["arch"], cfg["latent_dim"], device)
    loss_fn = get_loss_fn(cfg["loss"])

    opt_G = get_optimizer(cfg["optimizer"], G.parameters(),
                          cfg["lr"], wgan=(cfg["loss"].upper() == "WGAN"))
    opt_D = get_optimizer(cfg["optimizer"], D.parameters(),
                          cfg["lr"], wgan=(cfg["loss"].upper() == "WGAN"))

    # Fixed noise for consistent image grids across epochs
    fixed_noise = make_fixed_noise(64, cfg["latent_dim"], device)

    # Log model summary
    wandb.watch(G, log="all", log_freq=100)
    wandb.watch(D, log="all", log_freq=100)

    # ── Training loop ───────────────────────────
    best_g_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        g_loss, d_loss = train_one_epoch(
            G, D, train_loader, loss_fn,
            opt_G, opt_D,
            latent_dim = cfg["latent_dim"],
            device     = device,
            n_critic   = cfg["n_critic"],
        )

        # ── Generate image grid ─────────────────
        G.eval()
        with torch.no_grad():
            fake = G(fixed_noise)
            if fake.dim() == 2:                         # Vanilla → reshape
                fake = fake.view(-1, 1, 28, 28)
        G.train()

        grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))

        # ── Log to W&B ─────────────────────────
        wandb.log({
            "epoch"          : epoch,
            "g_loss"         : g_loss,
            "d_loss"         : d_loss,
            "generated_imgs" : wandb.Image(grid, caption=f"Epoch {epoch}"),
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{cfg['epochs']}]  "
                  f"G Loss: {g_loss:.4f}  D Loss: {d_loss:.4f}")

        # ── Save best generator ─────────────────
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            ckpt_path   = os.path.join(cfg["save_dir"], f"{run_name}_best_G.pth")
            torch.save(G.state_dict(), ckpt_path)

    # ── Save final checkpoint ────────────────────
    final_path = os.path.join(cfg["save_dir"], f"{run_name}_final_G.pth")
    torch.save(G.state_dict(), final_path)
    wandb.save(final_path)

    print(f"\n  [Done] Best G Loss: {best_g_loss:.4f}")
    print(f"  Checkpoint saved → {final_path}")

    run.finish()
    return G, D, best_g_loss
