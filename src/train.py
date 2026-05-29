"""
Training script for DDPM on CIFAR-10.
Usage: python src/train.py --config configs/default.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from src.dataset import get_data_loaders
from src.diffusion import Diffusion
from src.model import UNet
from src.utils import (
    set_seed,
    get_device,
    get_device_info,
    save_checkpoint,
    load_checkpoint,
    save_image_grid,
    plot_training_loss,
    visualize_diffusion_process,
    AverageMeter,
    ProgressMeter,
)


def train_one_epoch(
    model: nn.Module,
    diffusion: Diffusion,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    config: Config,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Denoising model
        diffusion: Diffusion process
        data_loader: Training data loader
        optimizer: Optimizer
        epoch: Current epoch number
        device: Device to train on
        config: Configuration object
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    loss_meter = AverageMeter("Loss")
    progress = ProgressMeter(
        len(data_loader),
        [loss_meter],
        prefix=f"Epoch [{epoch + 1}/{config.training.num_epochs}] ",
    )
    
    for batch_idx, x_0 in enumerate(data_loader):
        x_0 = x_0.to(device)
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, config.diffusion.num_timesteps, (batch_size,), device=device)
        
        # Forward diffusion: add noise
        x_t, noise = diffusion.q_sample(x_0, t)
        
        # Predict noise
        pred_noise = model(x_t, t)
        
        # Loss: MSE between predicted and actual noise
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
        
        optimizer.step()
        
        # Update loss meter
        loss_meter.update(loss.item(), batch_size)
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            progress.display(batch_idx + 1)
    
    return loss_meter.avg


def sample_images(
    model: nn.Module,
    diffusion: Diffusion,
    num_samples: int,
    device: torch.device,
    config: Config,
    epoch: int = None,
) -> torch.Tensor:
    """
    Generate sample images.
    
    Args:
        model: Denoising model
        diffusion: Diffusion process
        num_samples: Number of images to generate
        device: Device to generate on
        config: Configuration object
        epoch: Epoch number (for logging)
        
    Returns:
        Generated images tensor
    """
    model.eval()
    
    with torch.no_grad():
        shape = (num_samples, config.data.in_channels, config.data.image_size, config.data.image_size)
        images = diffusion.p_sample_loop(model, shape)
    
    return images


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = Config.load_yaml(args.config)
    
    # Create directories
    config.create_directories()
    
    # Set seed
    set_seed(config.training.seed)
    
    # Get device
    device = get_device(config.training.device)
    print(f"Device: {get_device_info()}")
    
    # Create data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(
        batch_size=config.data.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
        normalize_to=config.data.normalize_to,
        data_root="data",
    )
    print(f"Loaded {len(train_loader)} batches for training")
    print(f"Loaded {len(test_loader)} batches for testing")
    
    # Create model
    print("Creating U-Net model...")
    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        model_channels=config.model.model_channels,
        channel_mult=tuple(config.model.channel_mult),
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=tuple(config.model.attention_resolutions),
        num_heads=config.model.num_heads,
        num_head_channels=config.model.num_head_channels,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Create diffusion
    diffusion = Diffusion(
        num_timesteps=config.diffusion.num_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        clip_denoised=config.diffusion.clip_denoised,
        predict_noise=config.diffusion.predict_noise,
        device=device,
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if config.training.resume_checkpoint:
        start_epoch, best_loss = load_checkpoint(
            model,
            optimizer,
            config.training.resume_checkpoint,
            device,
        )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 50)
    
    losses = []
    
    try:
        for epoch in range(start_epoch, config.training.num_epochs):
            # Train for one epoch
            avg_loss = train_one_epoch(
                model,
                diffusion,
                train_loader,
                optimizer,
                epoch,
                device,
                config,
            )
            
            losses.append(avg_loss)
            
            # Save checkpoint
            if (epoch + 1) % config.training.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    avg_loss,
                    config.checkpoint_dir,
                    f"epoch_{epoch + 1}.pt",
                )
                
                # Save best checkpoint
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        avg_loss,
                        config.checkpoint_dir,
                        "best.pt",
                    )
            
            # Sample and save images
            if (epoch + 1) % config.training.sample_interval == 0:
                print(f"\nGenerating sample images at epoch {epoch + 1}...")
                
                num_samples = config.training.num_sample_batches * config.data.batch_size
                samples = sample_images(
                    model,
                    diffusion,
                    num_samples,
                    device,
                    config,
                    epoch,
                )
                
                sample_path = Path(config.sample_dir) / f"samples_epoch_{epoch + 1}.png"
                save_image_grid(
                    samples,
                    str(sample_path),
                    nrows=int(np.sqrt(num_samples)),
                    normalize_to=config.data.normalize_to,
                )
                
                # Visualize diffusion process with first sample
                if epoch == config.training.sample_interval - 1:
                    print("Visualizing diffusion process...")
                    x_0 = next(iter(train_loader))[0].to(device)
                    timesteps = [100, 250, 500, 750, 999]
                    viz_path = Path(config.plot_dir) / "diffusion_process.png"
                    visualize_diffusion_process(
                        x_0[0],
                        diffusion,
                        timesteps,
                        str(viz_path),
                        config.data.normalize_to,
                    )
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{config.training.num_epochs} - Loss: {avg_loss:.6f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    print("\nSaving final checkpoint...")
    save_checkpoint(
        model,
        optimizer,
        config.training.num_epochs - 1,
        losses[-1],
        config.checkpoint_dir,
        "final.pt",
    )
    
    # Plot training loss
    print("Plotting training loss...")
    plot_path = Path(config.plot_dir) / "training_loss.png"
    plot_training_loss(losses, str(plot_path), config.training.save_interval)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"Samples saved to: {config.sample_dir}")
    print(f"Plots saved to: {config.plot_dir}")


if __name__ == "__main__":
    main()
