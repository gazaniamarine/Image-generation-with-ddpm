"""
Smoke tests for the diffusion process and U-Net model.

These do not check image quality (that requires real training). They check
that the forward/reverse math is internally consistent and shape/NaN-safe,
which is cheap enough to run on every change and would have caught the
p_sample posterior-mean bug immediately.
"""
import torch

from src.diffusion import Diffusion
from src.model import UNet


def make_tiny_model():
    return UNet(
        in_channels=3,
        out_channels=3,
        model_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        num_heads=2,
    )


def test_q_sample_shape_and_range():
    diffusion = Diffusion(num_timesteps=50)
    x_0 = torch.rand(4, 3, 32, 32) * 2 - 1  # in [-1, 1]
    t = torch.randint(0, 50, (4,))

    x_t, noise = diffusion.q_sample(x_0, t)

    assert x_t.shape == x_0.shape
    assert noise.shape == x_0.shape
    assert torch.isfinite(x_t).all()

    # With zero noise, x_t should be exactly sqrt(alpha_cumprod_t) * x_0.
    t0 = torch.zeros(4, dtype=torch.long)
    x_t0, _ = diffusion.q_sample(x_0, t0, noise=torch.zeros_like(x_0))
    assert torch.allclose(x_t0, diffusion.schedule.sqrt_alphas_cumprod[0] * x_0, atol=1e-5)


def test_model_forward_shape():
    model = make_tiny_model()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 50, (2,))

    out = model(x, t)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_p_sample_shape_and_finite():
    model = make_tiny_model()
    diffusion = Diffusion(num_timesteps=50)

    x_t = torch.randn(2, 3, 32, 32)
    t = torch.full((2,), 25, dtype=torch.long)

    with torch.no_grad():
        x_prev = diffusion.p_sample(model, x_t, t)

    assert x_prev.shape == x_t.shape
    assert torch.isfinite(x_prev).all()


def test_p_sample_posterior_mean_matches_reference():
    """
    Regression test for the posterior-mean bug (mean must combine the
    *predicted x_0* and x_t, not x_t and the raw noise prediction).

    Using a model that always predicts zero noise, x_0_pred = x_t / sqrt(alpha_cumprod_t)
    in closed form, so the correct posterior mean can be computed independently
    and compared against p_sample's output (with t=0 to remove the stochastic
    noise term).
    """
    class ZeroNoiseModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    diffusion = Diffusion(num_timesteps=50, clip_denoised=False)
    model = ZeroNoiseModel()

    x_t = torch.randn(3, 3, 8, 8)
    t = torch.zeros(3, dtype=torch.long)  # t=0 -> no stochastic noise term added

    sqrt_alphas_cumprod = diffusion.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    coef1 = diffusion.schedule.posterior_mean_coef1[t].view(-1, 1, 1, 1)
    coef2 = diffusion.schedule.posterior_mean_coef2[t].view(-1, 1, 1, 1)

    expected_x0 = x_t / sqrt_alphas_cumprod
    expected_mean = coef1 * expected_x0 + coef2 * x_t

    with torch.no_grad():
        x_prev = diffusion.p_sample(model, x_t, t)

    assert torch.allclose(x_prev, expected_mean, atol=1e-4)
