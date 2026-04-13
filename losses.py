"""
losses.py - BCE, LSGAN, and WGAN loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss:
    """
    Standard Binary Cross-Entropy GAN loss.
    Uses label smoothing for the real samples (0.9 instead of 1.0)
    to improve training stability.
    """
    def __init__(self, label_smoothing: float = 0.9):
        self.criterion = nn.BCEWithLogitsLoss()
        self.real_label = label_smoothing

    def discriminator_loss(self, real_logits, fake_logits, device):
        real_labels = torch.full_like(real_logits, self.real_label, device=device)
        fake_labels = torch.zeros_like(fake_logits, device=device)
        d_real = self.criterion(real_logits, real_labels)
        d_fake = self.criterion(fake_logits, fake_labels)
        return (d_real + d_fake) / 2

    def generator_loss(self, fake_logits, device):
        real_labels = torch.ones_like(fake_logits, device=device)
        return self.criterion(fake_logits, real_labels)

    @property
    def name(self):
        return "BCE"

    @property
    def clip_weights(self):
        return False


class LSGANLoss:
    """
    Least Squares GAN loss (Mao et al., 2017).
    Uses MSE instead of BCE → avoids vanishing gradients near saturation.
    a=0 (fake), b=1 (real), c=1 (generator target)
    """
    def discriminator_loss(self, real_logits, fake_logits, device):
        d_real = 0.5 * torch.mean((real_logits - 1) ** 2)
        d_fake = 0.5 * torch.mean(fake_logits ** 2)
        return d_real + d_fake

    def generator_loss(self, fake_logits, device):
        return 0.5 * torch.mean((fake_logits - 1) ** 2)

    @property
    def name(self):
        return "LSGAN"

    @property
    def clip_weights(self):
        return False


class WGANLoss:
    """
    Wasserstein GAN loss (Arjovsky et al., 2017).
    Discriminator (critic) is NOT passed through sigmoid.
    Weight clipping enforces Lipschitz constraint.
    """
    def __init__(self, clip_value: float = 0.01):
        self.clip_value = clip_value

    def discriminator_loss(self, real_logits, fake_logits, device):
        # Critic maximises  E[D(real)] - E[D(fake)]
        # We minimise the negative
        return -torch.mean(real_logits) + torch.mean(fake_logits)

    def generator_loss(self, fake_logits, device):
        # Generator maximises E[D(fake)]
        return -torch.mean(fake_logits)

    @property
    def name(self):
        return "WGAN"

    @property
    def clip_weights(self):
        return True

    def clip(self, discriminator):
        for p in discriminator.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)


def get_loss_fn(loss_name: str):
    """Factory function — returns the requested loss object."""
    loss_name = loss_name.upper()
    if loss_name == "BCE":
        return BCELoss()
    elif loss_name == "LSGAN":
        return LSGANLoss()
    elif loss_name == "WGAN":
        return WGANLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from BCE, LSGAN, WGAN.")
