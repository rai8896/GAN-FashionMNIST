"""
models.py - Vanilla GAN and DCGAN architectures
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  VANILLA GAN
# ─────────────────────────────────────────────

class VanillaGenerator(nn.Module):
    """
    Fully-connected generator for Vanilla GAN.
    Input : latent vector z (latent_dim,)
    Output: flattened image (1×28×28 = 784)
    """
    def __init__(self, latent_dim: int = 100, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_dim),
            nn.Tanh(),          # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class VanillaDiscriminator(nn.Module):
    """
    Fully-connected discriminator for Vanilla GAN.
    Input : flattened image (784,)
    Output: scalar logit
    """
    def __init__(self, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ─────────────────────────────────────────────
#  DCGAN
# ─────────────────────────────────────────────

class DCGANGenerator(nn.Module):
    """
    Deep Convolutional Generator.
    Input : latent vector z (latent_dim, 1, 1)
    Output: image (1, 28, 28)

    Architecture (transposed convolutions):
    z(100) → 4×4 → 7×7 → 14×14 → 28×28
    """
    def __init__(self, latent_dim: int = 100, ngf: int = 64):
        super().__init__()
        # Project and reshape
        self.project = nn.Sequential(
            nn.Linear(latent_dim, ngf * 4 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.ngf = ngf

        self.net = nn.Sequential(
            # (ngf*4) × 4 × 4  →  (ngf*2) × 7 × 7
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=0),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            # (ngf*2) × 7 × 7  →  (ngf) × 14 × 14
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # (ngf) × 14 × 14  →  1 × 28 × 28
            nn.ConvTranspose2d(ngf, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.project(z)
        x = x.view(x.size(0), self.ngf * 4, 4, 4)
        return self.net(x)


class DCGANDiscriminator(nn.Module):
    """
    Deep Convolutional Discriminator.
    Input : image (1, 28, 28)
    Output: scalar logit
    """
    def __init__(self, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 1 × 28 × 28  →  ndf × 14 × 14
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf × 14 × 14  →  ndf*2 × 7 × 7
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*2 × 7 × 7  →  ndf*4 × 4 × 4
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*4 × 4 × 4  →  1 × 1 × 1
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  Weight Initialization
# ─────────────────────────────────────────────

def weights_init(m):
    """Apply DCGAN-style weight initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
