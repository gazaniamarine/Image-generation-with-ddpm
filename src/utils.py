"""
Utility functions for training, sampling, and visualization.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, List
import os


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """
    Get device for computation.
    
    Args:
        device: "auto" (use CUDA if available), "cuda", or "cpu"
        
    Returns:
        torch.device object
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)


def get_device_info() -> str:
    """Get information about available device."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        return f"CUDA available - {device_count} GPU(s): {device_name}"
    else:
        return "CUDA not available - using CPU"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    filepath = Path(checkpoint_dir) / filename
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    checkpoint_path: str = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Load checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        checkpoint_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Tuple of (epoch, loss)
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print("No checkpoint found")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float('inf'))
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch, loss


def unnormalize_images(images: torch.Tensor, normalize_to: str = "[-1, 1]") -> torch.Tensor:
    """
    Unnormalize images for visualization.
    
    Args:
        images: Normalized images tensor
        normalize_to: Normalization scheme used
        
    Returns:
        Unnormalized images in [0, 1]
    """
    if normalize_to == "[-1, 1]":
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2
    
    # Clamp to [0, 1]
    images = torch.clamp(images, 0, 1)
    
    return images


def save_image_grid(
    images: torch.Tensor,
    filepath: str,
    nrows: int = 8,
    normalize_to: str = "[-1, 1]",
):
    """
    Save images as a grid.
    
    Args:
        images: Tensor of shape (batch_size, channels, height, width)
        filepath: Path to save the image grid
        nrows: Number of rows in the grid
        normalize_to: Normalization scheme used
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Unnormalize for visualization
    images = unnormalize_images(images, normalize_to)
    
    # Create grid
    import torchvision.utils as vutils
    grid = vutils.make_grid(images, nrow=nrows, normalize=False)
    
    # Save
    vutils.save_image(grid, filepath)
    print(f"Image grid saved: {filepath}")


def plot_training_loss(
    losses: List[float],
    plot_path: str,
):
    """
    Plot training loss.

    Args:
        losses: List of per-epoch loss values (one entry per epoch)
        plot_path: Path to save plot
    """
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Loss plot saved: {plot_path}")


def visualize_diffusion_process(
    x_0: torch.Tensor,
    diffusion,
    timesteps: List[int],
    plot_path: str,
    normalize_to: str = "[-1, 1]",
):
    """
    Visualize forward diffusion process.
    
    Args:
        x_0: Original image tensor of shape (channels, height, width)
        diffusion: Diffusion object
        timesteps: List of timesteps to visualize
        plot_path: Path to save plot
        normalize_to: Normalization scheme used
    """
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    
    device = x_0.device
    x_0_batch = x_0.unsqueeze(0)  # Add batch dimension
    
    images = [x_0_batch.cpu()]
    titles = ["Original (t=0)"]
    
    # Generate random noise for consistency
    noise = torch.randn_like(x_0_batch)
    
    for t_val in timesteps:
        t = torch.tensor([t_val], device=device)
        x_t, _ = diffusion.q_sample(x_0_batch, t, noise=noise)
        images.append(x_t.cpu())
        titles.append(f"t={t_val}")
    
    # Unnormalize for visualization
    images = [unnormalize_images(img, normalize_to) for img in images]
    
    # Plot
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    for ax, img, title in zip(axes, images, titles):
        # Squeeze batch dimension and move to numpy
        img_np = img.squeeze(0).permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(title)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()
    
    print(f"Diffusion visualization saved: {plot_path}")


def save_generation_gif(
    images: List[torch.Tensor],
    gif_path: str,
    normalize_to: str = "[-1, 1]",
    duration: int = 50,
):
    """
    Save generation process as GIF.
    
    Args:
        images: List of image tensors
        gif_path: Path to save GIF
        normalize_to: Normalization scheme used
        duration: Duration of each frame in milliseconds
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available. Skipping GIF generation.")
        return
    
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Unnormalize and convert to PIL images
    pil_images = []
    for img_tensor in images:
        img_tensor = unnormalize_images(img_tensor, normalize_to)
        img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))
    
    # Save as GIF
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )
    
    print(f"Generation GIF saved: {gif_path}")


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress during training."""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
