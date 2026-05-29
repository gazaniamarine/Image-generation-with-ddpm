"""
Sampling script for generating images from a trained DDPM model.
Usage: python src/sample.py --checkpoint results/checkpoints/best.pt --num_samples 64
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from src.diffusion import Diffusion
from src.model import UNet
from src.utils import (
    set_seed,
    get_device,
    get_device_info,
    load_checkpoint,
    save_image_grid,
    save_generation_gif,
)


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Sample from trained DDPM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Save denoising process as GIF",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/samples",
        help="Output directory for generated images",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = Config.load_yaml(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Device: {get_device_info()}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading checkpoint from {args.checkpoint}")
    
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
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path=args.checkpoint, device=device)
    
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
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    model.eval()
    with torch.no_grad():
        shape = (args.num_samples, config.model.in_channels, config.data.image_size, config.data.image_size)
        
        if args.save_gif:
            print("Saving denoising process as GIF...")
            # Sample with return_all_timesteps to capture the denoising process
            all_images = diffusion.p_sample_loop(model, shape, return_all_timesteps=True)
            
            # Sample every N steps for GIF to keep file size reasonable
            num_steps = all_images.shape[1]
            step_interval = max(1, num_steps // config.sampling.num_gif_steps)
            selected_steps = list(range(0, num_steps, step_interval))
            
            # Take first sample
            selected_images = all_images[0, selected_steps]
            
            gif_path = Path(args.output_dir) / "generation_process.gif"
            save_generation_gif(
                [selected_images[i:i+1] for i in range(selected_images.shape[0])],
                str(gif_path),
                normalize_to=config.data.normalize_to,
            )
        else:
            # Just generate final images
            images = diffusion.p_sample_loop(model, shape)
        
        if not args.save_gif:
            images = diffusion.p_sample_loop(model, shape)
        else:
            # Extract final images from all_images
            images = all_images[:, -1]
    
    # Save images
    sample_path = Path(args.output_dir) / "samples.png"
    save_image_grid(
        images,
        str(sample_path),
        nrows=int(np.sqrt(args.num_samples)),
        normalize_to=config.data.normalize_to,
    )
    
    # Save individual images
    print("Saving individual samples...")
    individual_dir = Path(args.output_dir) / "individual"
    individual_dir.mkdir(exist_ok=True)
    
    for i in range(min(16, images.shape[0])):
        img_path = individual_dir / f"sample_{i:03d}.png"
        single_img = images[i:i+1]
        save_image_grid(
            single_img,
            str(img_path),
            nrows=1,
            normalize_to=config.data.normalize_to,
        )
    
    print("\n" + "=" * 50)
    print("Sampling completed!")
    print(f"Samples saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
