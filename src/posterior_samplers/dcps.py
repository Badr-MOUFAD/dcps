import tqdm
import torch
from typing import Callable, List

from posterior_samplers.diffusion_utils import (
    EpsilonNet,
    bridge_kernel_statistics,
    ddim,
    sample_bridge_kernel,
    ddim_statistics,
)
from utils.utils import display
import numpy as np


def ula(
    init_sample: torch.Tensor,
    pot_fn: Callable,
    epsilon_net: EpsilonNet,
    t: int,
    ula_steps: int,
    gamma: float = 1e-3,
):
    sample = init_sample.clone()
    n_samples = sample.shape[0]
    shape = (n_samples, *(1,) * len(sample.shape[1:]))
    acp_t = epsilon_net.alphas_cumprod[t]
    sample.requires_grad_()

    for _ in range(ula_steps):
        # sample.requires_grad_()
        eps_t = epsilon_net(sample, t)
        xhat_0 = (sample - (1 - acp_t) ** 0.5 * eps_t) / (acp_t**0.5)
        pot_val = pot_fn(sample, xhat_0)
        grad = torch.autograd.grad(pot_val, sample)[0]

        # with torch.no_grad():
        score = -eps_t / (1 - acp_t) ** 0.5
        grad = score + grad

        grad_norm = torch.norm(grad.reshape(n_samples, -1), dim=-1).reshape(*shape)
        step_size = gamma / (1 + (gamma * grad_norm))
        sample = (
            sample + step_size * grad + (2 * gamma) ** 0.5 * torch.randn_like(sample)
        )
        # sample = sample.detach()

    return sample.detach()


def ula_step(
    init_sample,
    epsilon_net,
    inverse_problem,
    t_obs,
    t_obs_last,
    ula_steps,
    gamma=1e-3,
    eta=1.0,
):
    scaled_obs, A, Amat, obs_std = inverse_problem

    def ipot(x, xhat_0):
        mean_taukItauknext, std_taukItauknext = ddim_statistics(
            x, epsilon_net, t_obs_last, t_obs, eta, e_t=xhat_0
        )
        return (
            -0.5
            * torch.norm(
                (scaled_obs - A(mean_taukItauknext)) / obs_std(mean_taukItauknext)
            )
            ** 2
        ).sum()

    return ula(
        init_sample,
        pot_fn=ipot,
        epsilon_net=epsilon_net,
        t=t_obs_last,
        ula_steps=ula_steps,
        gamma=gamma,
    )


def kl_div(
    curr_mean,
    curr_log_std,
    prior_mean,
    prior_std,
    epsilon_net,
    inverse_problem,
    t_prev,
    t_obs,
    eta,
):
    scaled_obs, A, Amat, obs_std = inverse_problem

    def kl_mvn(curr_mean, curr_log_std, prior_mean, prior_std):
        return 0.5 * (
            -2 * (curr_log_std - prior_std.log()).sum()
            + (2 * (curr_log_std - prior_std.log())).exp().sum()
            + torch.norm(curr_mean - prior_mean) ** 2 / (prior_std**2)
        )

    q_sample = curr_mean + curr_log_std.exp() * torch.randn_like(curr_mean)
    mean_bw_tau, std_bw_tau = ddim_statistics(q_sample, epsilon_net, t_prev, t_obs, eta)
    sample_taukl = mean_bw_tau + std_bw_tau * torch.randn_like(curr_mean)

    def pot_func(x):
        return -0.5 * torch.norm((scaled_obs - A(x)) / obs_std(x)) ** 2

    return (
        -pot_func(sample_taukl) + kl_mvn(curr_mean, curr_log_std, prior_mean, prior_std)
    ).sum()


def dcps_step(
    xt,
    epsilon_net,
    t,
    t_prev,
    t_obs,
    inverse_problem,
    optimizer,
    gradient_steps,
    learning_rate,
    eta,
    init_mean=None,
    init_logstd=None,
):
    """Implements the algorithm."""
    mean_tprevIt, std_tprevIt = ddim_statistics(xt, epsilon_net, t, t_prev, eta)

    mean = mean_tprevIt.clone() if init_mean is None else init_mean
    log_std = (
        torch.full(mean.shape, std_tprevIt.log().clone())
        if init_logstd is None
        else init_logstd
    )

    def kl(curr_mean, curr_log_std):
        return kl_div(
            curr_mean,
            curr_log_std,
            mean_tprevIt,
            std_tprevIt,
            epsilon_net,
            inverse_problem,
            t_prev,
            t_obs,
            eta,
        )

    kl_vals = []
    if optimizer == "SGD":
        for _ in range(gradient_steps):

            mean.requires_grad_(), log_std.requires_grad_()
            kldiv = kl(mean, log_std)
            mean_grad, logstd_grad = torch.autograd.grad(kldiv, (mean, log_std))

            mean = normalized_grad_step(mean, mean_grad, lr=learning_rate)
            log_std = normalized_grad_step(log_std, logstd_grad, lr=learning_rate)

    if optimizer == "Adam":
        mean, log_std = mean.requires_grad_(), log_std.requires_grad_()
        optim = torch.optim.Adam(lr=learning_rate, params=[mean, log_std])
        for _ in range(gradient_steps):
            optim.zero_grad()
            kld = kl(mean, log_std)
            kld.backward()
            optim.step()
            kl_vals.append(
                torch.norm(mean - mean_tprevIt, dim=-1).mean().cpu().detach()
            )

    return mean.detach(), log_std.detach(), kl_vals


def dcps(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    obs: torch.Tensor,
    A: torch.Tensor,
    obs_std: float,
    n_steps: int,
    obs_timesteps: List[int],
    ip_type: str = "linear",
    optimizer: str = "SGD",
    gradient_steps: int = 2,
    learning_rate: float = 1.0,
    langevin_steps=10,
    gamma: float = 1e-3,
    noise_type: str = "gaussian",
    poisson_rate: float = 0.1,
    display_im: bool = False,
    display_freq: int = 100,
    Amat=None,
    eta=1.0,
) -> torch.Tensor:
    x_tprev = initial_noise

    for idx, t_obs in enumerate(range(len(obs_timesteps) - 1, 0, -1)):
        timesteps = torch.linspace(
            obs_timesteps[t_obs - 1], obs_timesteps[t_obs], n_steps + 1
        ).int()
        t_obs, t_obs_last = timesteps[0], timesteps[-1]
        acp_t = epsilon_net.alphas_cumprod[t_obs]

        if ip_type == "linear":

            if noise_type == "gaussian":
                scaled_obs = (acp_t**0.5) * obs
                inverse_problem = (scaled_obs, A, Amat, lambda x: obs_std)

            elif noise_type == "poisson":
                rate = poisson_rate
                scaled_obs = (acp_t**0.5) * obs
                scaled_obs = rate * 127.5 * (scaled_obs + 1.0)
                std_fn = lambda x: (
                    rate * A(127.5 * (x + 1.0))
                    if t_obs.float() == 0.0
                    else scaled_obs.clip(1, 255).sqrt()
                )
                inverse_problem = (
                    scaled_obs,
                    lambda x: rate * A(127.5 * (x + 1.0)),
                    None,
                    lambda x: scaled_obs.clip(1, 255).sqrt(),
                )

        elif ip_type == "jpeg":
            im_shape = (1, 3, 256, 256)
            shifted_obs = (obs.reshape(im_shape) + 1.0) / 2.0
            interm_obs = acp_t.sqrt() * obs

            inverse_problem = (
                interm_obs,
                lambda x: A(x),
                None,
                lambda x: obs_std,
            )
            # display(interm_obs.detach().cpu())

        x_tprev = ula_step(
            x_tprev,
            epsilon_net,
            inverse_problem,
            t_obs,
            t_obs_last,
            ula_steps=langevin_steps,
            gamma=gamma,
            eta=eta,
        )

        last_idx = 1 if t_obs == 0 else 0
        kl_value_taukl = []

        for i in tqdm.tqdm(
            range(len(timesteps) - 1, last_idx, -1),
            desc=f"step {idx + 1} / {len(obs_timesteps) - 1}",
        ):
            t, t_prev = timesteps[i], timesteps[i - 1]

            if idx > 100:
                init_mean, init_logstd = mean, log_std
            else:
                init_mean, init_logstd = None, None

            mean, log_std, kl_vals = dcps_step(
                xt=x_tprev,
                epsilon_net=epsilon_net,
                t=t,
                t_prev=t_prev,
                t_obs=t_obs,
                inverse_problem=inverse_problem,
                optimizer=optimizer,
                gradient_steps=gradient_steps,
                learning_rate=learning_rate,
                eta=eta,
                init_mean=init_mean,
                init_logstd=init_logstd,
            )
            x_tprev = mean + log_std.exp() * torch.randn_like(mean)
            kl_value_taukl += kl_vals

            if display_im and i % display_freq == 0:
                img = epsilon_net.predict_x0(x_tprev[[0]], t_prev)
                display(img)

    return x_tprev


@torch.no_grad()
def normalized_grad_step(var: torch.Tensor, var_grad: torch.Tensor, lr: float):
    """Apply a normalized gradient step on ``var``.

    Formula of the update::

        var = var - (lr / norm(grad)) * grad

    Note
    ----
    ``var`` must a be a leaf tensor with ``requires_grad=True``
    """
    # NOTE this is the eps used in Adam solver to prevent denominator from being zero
    eps = 1e-8
    n_samples = var.shape[0]
    shape = (n_samples, *(1,) * len(var.shape[1:]))

    grad_norm = torch.norm(var_grad.reshape(n_samples, -1), dim=-1).reshape(*shape)

    return var - (lr / (eps + grad_norm)) * var_grad
