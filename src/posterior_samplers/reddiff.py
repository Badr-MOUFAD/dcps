import torch
from tqdm import tqdm
from math import prod
from typing import Callable
from torch.func import grad
from utils.utils import display
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.diffusion_utils import ddim_statistics, sample_bridge_kernel


def red_diff(initial_mean, log_pot, tau, epsilon_net, n_gradient_steps, lr):

    acp_tau = epsilon_net.alphas_cumprod[tau]
    mean, log_std = initial_mean.requires_grad_(), torch.tensor(
        1 - acp_tau
    ).log() * torch.ones_like(initial_mean)
    optimizer = torch.optim.Adam(lr=lr, params=[mean, log_std])

    for i in range(n_gradient_steps):
        optimizer.zero_grad()
        t = torch.randint(tau, epsilon_net.timesteps[-1], (1,))
        acp_t = epsilon_net.alphas_cumprod[t]
        noise = torch.randn_like(mean)
        xt = (
            acp_t.sqrt() * mean
            + (1 - acp_t + acp_t * (2 * log_std).exp()).sqrt() * noise
        )
        lambda_t = 50.0
        eps = epsilon_net(xt, t)
        loss = log_pot(mean) + lambda_t * ((eps.detach() - noise) * mean).sum()
        loss.backward()
        optimizer.step()

    return mean + log_std.exp() * torch.randn_like(mean)


def ula_init(initial_sample, log_pot, tau, epsilon_net, ula_steps, lr=1e-1):

    n_samples = initial_sample.shape[0]
    shape = (n_samples, *(1,) * len(initial_sample.shape[1:]))
    acp_tau = epsilon_net.alphas_cumprod[tau]
    grad_logpot = grad(log_pot)
    sample = initial_sample

    for i in range(ula_steps):
        grad_log_post = grad_logpot(sample) + epsilon_net.score(sample, tau)
        grad_norm = torch.norm(grad_log_post.reshape(n_samples, -1), dim=-1).reshape(
            shape
        )
        G_gamma = lr * grad_log_post / (1 + lr * grad_norm)
        sample = sample + G_gamma + (2 * lr) ** 0.5 * torch.randn_like(sample)

        if i % 100 == 0:
            display(epsilon_net.predict_x0(sample, tau))

    return sample
