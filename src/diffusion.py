"""
Diffusion process for DDPM.
Handles forward diffusion (adding noise) and reverse diffusion (denoising).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class DiffusionSchedule:
    """
    Manages the noise schedule for diffusion process.
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize diffusion schedule.
        
        Args:
            num_timesteps: Number of diffusion steps
            beta_schedule: Type of schedule ("linear" or "cosine")
            beta_start: Starting beta value for linear schedule
            beta_end: Ending beta value for linear schedule
            device: Device to place tensors on
        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.device = device
        
        # Compute betas based on schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps, device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        # Useful for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Useful for posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    @staticmethod
    def _cosine_beta_schedule(num_timesteps: int, device: torch.device) -> torch.Tensor:
        """
        Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        s = 0.008
        steps = torch.arange(num_timesteps + 1, device=device)
        alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def get_sqrt_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Get sqrt(alpha_cumprod) for timesteps t."""
        return self.sqrt_alphas_cumprod[t]
    
    def get_sqrt_one_minus_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Get sqrt(1 - alpha_cumprod) for timesteps t."""
        return self.sqrt_one_minus_alphas_cumprod[t]


class Diffusion:
    """
    Main diffusion class handling forward and reverse processes.
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_denoised: bool = True,
        predict_noise: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize diffusion process.
        
        Args:
            num_timesteps: Number of diffusion steps
            beta_schedule: Type of schedule ("linear" or "cosine")
            beta_start: Starting beta value for linear schedule
            beta_end: Ending beta value for linear schedule
            clip_denoised: Clip denoised output to [-1, 1]
            predict_noise: Predict noise instead of mean
            device: Device to place tensors on
        """
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised
        self.predict_noise = predict_noise
        self.device = device
        
        self.schedule = DiffusionSchedule(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward diffusion process: add noise to x_0 at timestep t.
        
        q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        
        Args:
            x_0: Clean images (batch_size, channels, height, width)
            t: Timesteps (batch_size,)
            noise: Optional pre-sampled noise
            
        Returns:
            Tuple of (x_t, noise) where:
            - x_t: Noisy images at timestep t
            - noise: The noise added (can be used as target for noise prediction)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get coefficients for this timestep
        # Reshape for broadcasting: (batch_size, 1, 1, 1)
        sqrt_alphas_cumprod = self.schedule.get_sqrt_alphas_cumprod(t)
        sqrt_one_minus_alphas_cumprod = self.schedule.get_sqrt_one_minus_alphas_cumprod(t)
        
        # Reshape for proper broadcasting
        batch_size = x_0.shape[0]
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(batch_size, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(batch_size, 1, 1, 1)
        
        # Compute x_t
        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        
        return x_t, noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: denoise x_t to get x_{t-1}.
        
        Args:
            model: Denoising model
            x_t: Noisy image at timestep t
            t: Current timestep
            clip_denoised: Whether to clip denoised output
            
        Returns:
            x_{t-1}: Denoised image
        """
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        batch_size = x_t.shape[0]

        # Predict noise
        pred_noise = model(x_t, t)

        # Recover predicted x_0 from the noise prediction:
        # x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * eps) / sqrt(alpha_cumprod_t)
        sqrt_alphas_cumprod = self.schedule.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod * pred_noise) / sqrt_alphas_cumprod

        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        # Posterior mean: q(x_{t-1} | x_t, x_0) = coef1 * x_0 + coef2 * x_t
        coef1 = self.schedule.posterior_mean_coef1[t].view(batch_size, 1, 1, 1)
        coef2 = self.schedule.posterior_mean_coef2[t].view(batch_size, 1, 1, 1)
        mean = coef1 * x_0_pred + coef2 * x_t

        # Add noise for all steps except the last
        if t.min().item() > 0:
            posterior_var = self.schedule.posterior_variance[t].view(batch_size, 1, 1, 1)
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(posterior_var) * noise
        else:
            x_prev = mean

        return x_prev
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Generate images by sampling from the reverse diffusion process.
        
        Args:
            model: Denoising model
            shape: Shape of images to generate (batch_size, channels, height, width)
            return_all_timesteps: If True, return all intermediate steps
            
        Returns:
            Generated images or all intermediate steps
        """
        # Start from pure Gaussian noise
        img = torch.randn(shape, device=self.device)
        
        if return_all_timesteps:
            timesteps = []
        
        # Reverse diffusion process: go from T to 0
        for t in reversed(range(0, self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t_batch)
            
            if return_all_timesteps:
                timesteps.append(img.cpu())
        
        if return_all_timesteps:
            return torch.stack(timesteps, dim=1)
        
        return img


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from 1D array and broadcast to shape.
    
    Args:
        arr: 1D tensor
        timesteps: Indices to extract
        broadcast_shape: Shape to broadcast to
        
    Returns:
        Extracted and broadcasted tensor
    """
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
