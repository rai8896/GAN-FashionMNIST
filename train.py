"""
run_experiments.py
==================
Runs all 18 experiment combinations:
  2 architectures  ×  3 loss functions  ×  3 optimizers

Usage:
    python run_experiments.py                        # all 18
    python run_experiments.py --arch dcgan           # only DCGAN experiments
    python run_experiments.py --arch vanilla --loss bce --optimizer adam  # single run
"""

import argparse
import itertools
import json
import os
from datetime import datetime

from trainer import train_experiment


# ─────────────────────────────────────────────
#  All experiment combinations
# ─────────────────────────────────────────────

ARCHITECTURES = ["vanilla", "dcgan"]
LOSS_FUNCTIONS = ["bce", "lsgan", "wgan"]
OPTIMIZERS     = ["adam", "rmsprop", "sgd"]


def run_all_experiments(args):
    """Run all (or filtered) combinations and collect results."""

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build list of configs to run
    arch_list = [args.arch]        if args.arch      else ARCHITECTURES
    loss_list = [args.loss]        if args.loss      else LOSS_FUNCTIONS
    opt_list  = [args.optimizer]   if args.optimizer else OPTIMIZERS

    combos = list(itertools.product(arch_list, loss_list, opt_list))
    total  = len(combos)

    print(f"\n{'#'*60}")
    print(f"  Total experiments to run: {total}")
    print(f"  Epochs per run          : {args.epochs}")
    print(f"{'#'*60}\n")

    for idx, (arch, loss, opt) in enumerate(combos, 1):
        print(f"\n[{idx}/{total}] Starting: arch={arch}, loss={loss}, optimizer={opt}")

        config = {
            "arch"       : arch,
            "loss"       : loss,
            "optimizer"  : opt,
            "latent_dim" : args.latent_dim,
            "epochs"     : args.epochs,
            "batch_size" : args.batch_size,
            "lr"         : args.lr,
            "augment"    : args.augment,
            "data_dir"   : args.data_dir,
            "save_dir"   : args.save_dir,
        }

        try:
            G, D, best_g_loss = train_experiment(
                config,
                project_name=args.wandb_project
            )
            status = "success"
        except Exception as e:
            print(f"  [ERROR] Experiment failed: {e}")
            best_g_loss = None
            status = f"failed: {str(e)}"

        results.append({
            "arch"       : arch,
            "loss"       : loss,
            "optimizer"  : opt,
            "best_g_loss": best_g_loss,
            "status"     : status,
        })

    # ── Save summary ────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    summary_path = os.path.join(args.save_dir, f"results_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Arch':<10} {'Loss':<8} {'Optimizer':<12} {'Best G Loss':<14} Status")
    print(f"  {'-'*58}")
    for r in results:
        g_loss_str = f"{r['best_g_loss']:.4f}" if r['best_g_loss'] is not None else "N/A"
        print(f"  {r['arch']:<10} {r['loss']:<8} {r['optimizer']:<12} {g_loss_str:<14} {r['status']}")

    print(f"\n  Results saved → {summary_path}")
    return results


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run GAN experiments on Fashion-MNIST")

    # Filtering (leave blank to run all)
    p.add_argument("--arch",       type=str, default=None,
                   choices=["vanilla", "dcgan"],
                   help="Architecture (default: all)")
    p.add_argument("--loss",       type=str, default=None,
                   choices=["bce", "lsgan", "wgan"],
                   help="Loss function (default: all)")
    p.add_argument("--optimizer",  type=str, default=None,
                   choices=["adam", "rmsprop", "sgd"],
                   help="Optimizer (default: all)")

    # Hyperparameters
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--latent_dim",  type=int,   default=100)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--augment",     action="store_true",
                   help="Enable horizontal flip augmentation")

    # Paths
    p.add_argument("--data_dir",        type=str, default="./data")
    p.add_argument("--save_dir",        type=str, default="./checkpoints")
    p.add_argument("--wandb_project",   type=str, default="GAN-FashionMNIST")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args)
