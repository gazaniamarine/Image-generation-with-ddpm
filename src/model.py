"""
U-Net architecture for DDPM denoising.
A simplified U-Net suitable for 32x32 CIFAR-10 images.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            time: Timestep tensor of shape (batch_size,)
            
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Compute frequencies
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
        )
        
        # Compute embedding
        angles = time[:, None].float() * frequencies[None, :]
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embed_dim: Dimension of time embedding
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First convolution
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Time embedding projection
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )
        
        # Second convolution
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            time_emb: Time embedding of shape (batch_size, time_embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # First convolution
        h = self.conv1(x)
        
        # Add time embedding
        time_emb_out = self.time_emb(time_emb)
        h = h + time_emb_out[:, :, None, None]
        
        # Second convolution
        h = self.conv2(h)
        
        # Add skip connection
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_head_channels: int = -1,
    ):
        """
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
            num_head_channels: Channels per head (-1 for automatic)
        """
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        if num_head_channels == -1:
            self.head_channels = channels // num_heads
        else:
            self.head_channels = num_head_channels
        
        # Normalize and project
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Reshape to (batch_size, height*width, channels)
        h = h.view(batch_size, channels, -1).transpose(1, 2)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_channels).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_channels).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_channels).transpose(1, 2)
        
        # Compute attention
        scale = self.head_channels ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, channels)
        
        # Project
        out = self.proj(out)
        
        # Reshape back to (batch_size, channels, height, width)
        out = out.transpose(1, 2).view(batch_size, channels, height, width)
        
        # Add skip connection
        return out + x


class Downsample(nn.Module):
    """Downsampling block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for DDPM denoising on 32x32 images.
    Simplified architecture suitable for CIFAR-10.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        num_heads: int = 4,
        num_head_channels: int = -1,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            model_channels: Base channel count
            channel_mult: Channel multipliers for each resolution
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions to apply attention
            num_heads: Number of attention heads
            num_head_channels: Channels per attention head
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.num_levels = len(channel_mult)
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Build encoder (downsampling) blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        
        for level in range(self.num_levels):
            ch_mult = channel_mult[level]
            out_ch = model_channels * ch_mult
            
            # Residual blocks at this resolution
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch, time_embed_dim, dropout))
                
                # Attention at specified resolutions
                if 32 // (2 ** level) in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(out_ch, num_heads, num_head_channels))
                
                ch = out_ch
            
            # Downsample between levels (but not after the last level)
            if level < self.num_levels - 1:
                self.down_blocks.append(Downsample(ch))
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.middle_blocks.append(ResidualBlock(ch, ch, time_embed_dim, dropout))
        
        self.middle_blocks.append(AttentionBlock(ch, num_heads, num_head_channels))
        
        for _ in range(num_res_blocks):
            self.middle_blocks.append(ResidualBlock(ch, ch, time_embed_dim, dropout))
        
        # Build decoder (upsampling) blocks
        self.up_blocks = nn.ModuleList()
        
        # Decoder stages correspond to all encoder levels except the deepest one
        for level in reversed(range(self.num_levels - 1)):
            ch_mult = channel_mult[level]
            skip_ch = model_channels * ch_mult
            
            # Upsample before concatenating the corresponding skip connection
            self.up_blocks.append(Upsample(ch))
            
            # Residual blocks at this resolution
            for block_idx in range(num_res_blocks):
                in_ch = ch + skip_ch if block_idx == 0 else skip_ch
                self.up_blocks.append(ResidualBlock(in_ch, skip_ch, time_embed_dim, dropout))
                
                if 32 // (2 ** level) in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(skip_ch, num_heads, num_head_channels))
            
            ch = skip_ch
        
        # Output convolution
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            t: Timesteps of shape (batch_size,)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Time embedding
        time_emb = self.time_embed(t)
        
        # Input convolution
        h = self.conv_in(x)
        
        # Encoder path with skip connections
        skips = []
        
        for block in self.down_blocks:
            if isinstance(block, Downsample):
                skips.append(h)
                h = block(h)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else:  # ResidualBlock
                h = block(h, time_emb)
        
        # Middle blocks
        for block in self.middle_blocks:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, time_emb)
        
        # Decoder path with skip connections
        for block in self.up_blocks:
            if isinstance(block, Upsample):
                h = block(h)
                h = torch.cat([h, skips.pop()], dim=1)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else:  # ResidualBlock
                h = block(h, time_emb)
        
        # Output convolution
        return self.out(h)
