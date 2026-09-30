"""VP (variance-preserving) noise levels, shared by `backend_sd2` and the `sd2` flavour of `dummy`.

A level is (t, alpha, sigma) with x_t = alpha * x0 + sigma * eps, alpha = sqrt(alphas_cumprod[t]), sigma = sqrt(1 - alphas_cumprod[t]).
`noise_frac = sigma / (alpha + sigma)` in [0, 1] is the mixing weight of the noise: it is 1 at the pure-noise end and 0 at the clean end,
which is exactly what `sigma` is for a flow-matching model, so every relay threshold ("field:0.9") means the same thing on both.
Stepping from one level to the next with `alpha' x0_hat + sigma' eps_hat` is the DDIM (eta = 0) update."""
from collections import namedtuple
import numpy as np

class VPLevel(namedtuple("VPLevel", "t alpha sigma")):
    __slots__ = ()
    @property
    def noise_frac(self): return float(self.sigma / (self.alpha + self.sigma))

CLEAN = VPLevel(-1, 1.0, 0.0)                       # the final level: x = x0_hat

def scaled_linear_alphas_cumprod(n=1000, beta_start=0.00085, beta_end=0.012):
    """The Stable Diffusion 'scaled_linear' beta schedule, as numpy (the real backend reads the schedule off its own scheduler)."""
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)

def ddim_timesteps(n_train, steps, offset=1):
    """diffusers' DDIMScheduler ('leading' spacing, steps_offset=1 for Stable Diffusion): n_train // steps stride, descending."""
    stride = n_train // steps
    return (np.arange(0, steps) * stride)[::-1].astype(int) + offset

def ddim_levels(alphas_cumprod, steps, offset=1):
    """`steps` DDIM levels from a 1000-step training schedule, plus the final clean level."""
    acp = np.asarray(alphas_cumprod, np.float64); ts = ddim_timesteps(len(acp), steps, offset)
    return [VPLevel(int(t), float(np.sqrt(acp[int(t)])), float(np.sqrt(1.0 - acp[int(t)]))) for t in ts] + [CLEAN]

def nearest_t(alphas_cumprod, noise_frac):
    """The discrete training timestep whose noise_frac is closest to `noise_frac` (used to place the probe's t on a VP schedule)."""
    acp = np.asarray(alphas_cumprod, np.float64); a = np.sqrt(acp); s = np.sqrt(1.0 - acp)
    return int(np.argmin(np.abs(s / (a + s) - float(noise_frac))))
