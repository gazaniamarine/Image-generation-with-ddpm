"""
Configuration module for DDPM training and sampling.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import yaml
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "cifar10"  # Dataset name
    image_size: int = 32  # Image size (CIFAR-10 is 32x32)
    batch_size: int = 128  # Batch size for training
    num_workers: int = 4  # Number of workers for DataLoader
    normalize_to: str = "[-1, 1]"  # Normalize images to [-1, 1] or [0, 1]


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "unet"  # Type of model (only unet supported for now)
    in_channels: int = 3  # RGB images
    out_channels: int = 3  # Predict 3 channels
    model_channels: int = 128  # Base channel count
    channel_mult: tuple = (1, 2, 2, 2)  # Channel multipliers for each resolution
    num_res_blocks: int = 2  # Residual blocks per resolution
    attention_resolutions: tuple = (16,)  # Resolutions to apply attention
    num_heads: int = 4  # Number of attention heads
    num_head_channels: int = -1  # Automatic head channels if -1


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""
    num_timesteps: int = 1000  # Number of diffusion steps
    beta_schedule: str = "linear"  # "linear" or "cosine"
    beta_start: float = 0.0001  # Starting beta value for linear schedule
    beta_end: float = 0.02  # Ending beta value for linear schedule
    clip_denoised: bool = True  # Clip denoised output to [-1, 1]
    predict_noise: bool = True  # Predict noise instead of mean


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100  # Number of training epochs
    learning_rate: float = 1e-4  # Learning rate
    weight_decay: float = 0.0  # Weight decay for optimizer
    ema_decay: float = 0.9999  # Exponential moving average decay
    use_ema: bool = False  # Whether to use EMA (optional for beginners)
    gradient_clip: float = 1.0  # Gradient clipping value
    save_interval: int = 10  # Save checkpoint every N epochs
    sample_interval: int = 5  # Generate samples every N epochs
    num_sample_batches: int = 4  # Number of batches to sample (4 * batch_size images)
    device: str = "auto"  # "auto", "cuda", or "cpu"
    seed: int = 42  # Random seed for reproducibility
    resume_checkpoint: Optional[str] = None  # Path to checkpoint to resume from


@dataclass
class SamplingConfig:
    """Sampling configuration."""
    num_samples: int = 64  # Number of samples to generate
    checkpoint_path: str = "results/checkpoints/best.pt"  # Path to checkpoint
    save_gif: bool = True  # Save denoising process as GIF
    num_gif_steps: int = 50  # Number of steps to save for GIF
    device: str = "auto"  # "auto", "cuda", or "cpu"
    seed: int = 42  # Random seed


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    diffusion: DiffusionConfig = None
    training: TrainingConfig = None
    sampling: SamplingConfig = None
    
    # Paths
    results_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    sample_dir: str = "results/samples"
    plot_dir: str = "results/plots"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.diffusion is None:
            self.diffusion = DiffusionConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.sampling is None:
            self.sampling = SamplingConfig()
    
    @staticmethod
    def load_yaml(yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return Config.from_dict(config_dict)
    
    @staticmethod
    def from_dict(config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        data = DataConfig(**config_dict.get('data', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        diffusion = DiffusionConfig(**config_dict.get('diffusion', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        sampling = SamplingConfig(**config_dict.get('sampling', {}))
        
        return Config(
            data=data,
            model=model,
            diffusion=diffusion,
            training=training,
            sampling=sampling,
            results_dir=config_dict.get('results_dir', 'results'),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'results/checkpoints'),
            sample_dir=config_dict.get('sample_dir', 'results/samples'),
            plot_dir=config_dict.get('plot_dir', 'results/plots'),
        )
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'diffusion': asdict(self.diffusion),
            'training': asdict(self.training),
            'sampling': asdict(self.sampling),
            'results_dir': self.results_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'sample_dir': self.sample_dir,
            'plot_dir': self.plot_dir,
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def create_directories(self):
        """Create all necessary directories."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
