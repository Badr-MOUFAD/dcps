from typing import List

import torch
from torch.func import grad
from posterior_samplers.diffusion_utils import (
    EpsilonNet,
    ddim_step,
    EpsilonNetMCGD,
)
import tqdm
from ddrm.functions.denoising import efficient_generalized_steps


from mcg_diff.sgm import ScoreModel
from mcg_diff.particle_filter import mcg_diff
from mcg_diff.scripts.viz_gaussian import get_optimal_timesteps_from_singular_values


class EpsilonNetDDRM(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        t = t.to(int)
        return self.unet(x, t)


class EpsilonNetDPS(EpsilonNet):
    def __init__(self, net, alphas_cumprod, pot_func, timesteps):
        super().__init__(net, alphas_cumprod, timesteps)
        self.pot_func = pot_func
        self.base_epsnet = EpsilonNet(net, alphas_cumprod, timesteps)

    def forward(self, x, t):
        shape = (x.shape[0], *(1,) * len(x.shape[1:]))
        grad_int_pot_est = lambda x, t: grad(
            lambda z: self.pot_func(self.base_epsnet.predict_x0(z, t)).sum()
        )(x)
        grad_lklhd = grad_int_pot_est(x, t)
        grad_norm = torch.norm(grad_lklhd.reshape(x.shape[0], -1), dim=-1).reshape(
            *shape
        )
        grad_lklhd = (1 / grad_norm) * grad_lklhd
        return -((1 - self.alphas_cumprod[t]) ** 0.5) * grad_lklhd + self.base_epsnet(
            x, t
        )


# def ula(
#     init_sample: torch.Tensor, grad_func: Callable, ula_steps: int, gamma: float = 1e-3
# ):
#     sample = init_sample
#     n_samples = sample.shape[0]
#     shape = (n_samples, *(1,) * len(sample.shape[1:]))
#     for step in range(ula_steps):
#         grad = grad_func(sample)
#         grad_norm = torch.norm(grad.reshape(n_samples, -1), dim=-1).reshape(*shape)
#         step_size = gamma / (1 + (gamma * grad_norm))
#         sample = (
#             sample + step_size * grad + (2 * gamma) ** 0.5 * torch.randn_like(sample)
#         )
#     return sample


# def ula_posterior(
#     init_sample: torch.Tensor,
#     epsilon_net: EpsilonNet,
#     timesteps: List[int],
#     obs: torch.Tensor,
#     A: torch.Tensor,
#     obs_std: torch.Tensor,
#     gamma: float = 1e-3,
# ):
#     def pot(x, t):
#         alpha_cum_t = epsilon_net.alphas_cumprod[t]
#         obs_t = (alpha_cum_t**0.5) * obs
#         # std_t = (1 - alpha_cum_t + obs_std) ** .5
#         std_t = obs_std
#         return -0.5 * torch.norm(obs_t - x @ A.T, dim=-1) ** 2.0 / (std_t**2.0)

#     pot_grad = lambda x, t: grad(lambda z: pot(z, t).sum())(x)
#     sample = init_sample

#     for t_idx in tqdm.tqdm(range(len(timesteps) - 1, 1, -1)):
#         t = timesteps[t_idx]
#         sample = ula(
#             init_sample=sample,
#             grad_func=lambda x: pot_grad(x, t) + epsilon_net.score(x, t),
#             ula_steps=1,
#             gamma=gamma,
#         )

#     return sample


def ddrm(initial_noise, unet, inverse_problem, timesteps, alphas_cumprod, device):
    obs, A, std = inverse_problem
    ddrm_timesteps = timesteps.clone()
    ddrm_timesteps[-1] = ddrm_timesteps[-1] - 1
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    ddrm_samples = efficient_generalized_steps(
        x=initial_noise,
        b=betas,
        seq=ddrm_timesteps.cpu(),
        model=EpsilonNetDDRM(unet=unet),
        y_0=obs[None, ...].to(device),
        H_funcs=A,
        sigma_0=std,
        etaB=1.0,
        etaA=0.85,
        etaC=1.0,
        device=device,
        classes=None,
        cls_fn=None,
    )
    return ddrm_samples[0][-1]


def dps(
    initial_noise,
    inverse_problem,
    epsilon_net,
    gamma=1.0,
    eta=1.0,
    noise_type="gaussian",
    poisson_rate=1e-1,
):
    obs, A, std = inverse_problem
    shape = (initial_noise.shape[0], *(1,) * len(initial_noise.shape[1:]))
    if noise_type == "gaussian":
        pot_func = lambda x: -torch.norm(obs.reshape(1, -1) - A(x)) ** 2.0
        error = lambda x: torch.norm(obs.reshape(1, -1) - A(x), dim=-1)
    elif noise_type == "poisson":
        rate = poisson_rate
        obs = rate * (obs.reshape(1, -1) + 1.0) / 2.0
        pot_func = lambda x: -(
            torch.norm((obs - rate * A((x + 1.0) / 2.0)) / (obs + 1e-3).sqrt()) ** 2.0
        )
        error = lambda x: torch.norm(obs - rate * A((x + 1.0) / 2.0), dim=-1)

    sample = initial_noise
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_norm = error(e_t).reshape(*shape)
        pot_val = pot_func(e_t)
        grad_pot = torch.autograd.grad(pot_val, sample)[0]
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=epsilon_net.timesteps[i],
            t_prev=epsilon_net.timesteps[i - 1],
            eta=eta,
            e_t=e_t,
        ).detach()
        # grad_norm = torch.norm(grad_lklhd.reshape(sample.shape[0], -1), dim=-1).reshape(*shape)
        grad_pot = gamma * grad_pot / grad_norm
        sample = sample + grad_pot

    sample.requires_grad_()
    grad_lklhd = torch.autograd.grad(pot_func(sample), sample)[0]
    grad_norm = torch.norm(grad_lklhd.reshape(sample.shape[0], -1), dim=-1).reshape(
        *shape
    )
    grad_lklhd = (gamma / grad_norm) * grad_lklhd

    return (
        epsilon_net.predict_x0(sample, epsilon_net.timesteps[1]) + grad_lklhd
    ).detach()


def pgdm_svd(initial_noise, epsilon_net, obs, H_func, std_obs, eta=1.0):
    """
    obs = D^{-1} U^T y
    """
    Ut_y, diag = H_func.Ut(obs), H_func.singulars()

    def pot_fn(x, t):
        rsq_t = 1 - epsilon_net.alphas_cumprod[t]
        diag_cov = diag**2 + (std_obs**2 / rsq_t)
        return (
            -0.5
            * torch.norm((Ut_y - diag * x[:, : diag.shape[0]]) / diag_cov.sqrt()) ** 2.0
        )

    sample = initial_noise.reshape(initial_noise.shape[0], -1)
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = sample.requires_grad_()
        xhat_0 = epsilon_net.predict_x0(sample, t)
        acp_t, acp_tprev = (
            torch.tensor([epsilon_net.alphas_cumprod[t]]),
            torch.tensor([epsilon_net.alphas_cumprod[t_prev]]),
        )
        # grad_pot = grad_pot_fn(sample, t)
        grad_pot = pot_fn(xhat_0, t)
        grad_pot = torch.autograd.grad(grad_pot, sample)[0]
        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=xhat_0
        ).detach()
        sample += acp_tprev.sqrt() * acp_t.sqrt() * grad_pot

    return (
        H_func.V(epsilon_net.predict_x0(sample, epsilon_net.timesteps[1]))
        .reshape(initial_noise.shape)
        .detach()
    )


def pgdm_jpeg(initial_noise, epsilon_net, obs, H_func, eta=1.0):
    """
    obs corresponds to the decoding
    """

    def pot_fn(x, t):
        rsq_t = 1 - epsilon_net.alphas_cumprod[t]
        diff = (obs - H_func.H(x)).detach()
        return (diff * x.reshape(x.shape[0], -1)).sum()

    sample = initial_noise
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = sample.requires_grad_()
        xhat_0 = epsilon_net.predict_x0(sample, t)
        acp_t, acp_tprev = (
            torch.tensor([epsilon_net.alphas_cumprod[t]]),
            torch.tensor([epsilon_net.alphas_cumprod[t_prev]]),
        )
        # grad_pot = grad_pot_fn(sample, t)
        grad_pot = pot_fn(xhat_0, t)
        grad_pot = torch.autograd.grad(grad_pot, sample)[0]
        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=xhat_0
        ).detach()
        sample += acp_t.sqrt() * grad_pot  # * acp_tprev.sqrt()

    return epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])


def mcgdiff(
    initial_noise,
    epsilon_net,
    obs,
    H_func,
    coordinates_mask,
    std_obs,
    device,
    eta=1.0,
    n_return=1,
):
    Ut_y, diag = H_func.Ut(obs).flatten(), H_func.singulars()
    # the first dim is batch_size
    dim_x = initial_noise.shape[1:]

    score_model = ScoreModel(
        net=torch.nn.DataParallel(
            EpsilonNetMCGD(
                H_func, epsilon_net.net, dim=initial_noise.shape[1:]
            ).requires_grad_(False),
            device_ids=[device],
        ),
        alphas_cumprod=epsilon_net.alphas_cumprod,
        device=device,
    )
    adapted_timesteps = get_optimal_timesteps_from_singular_values(
        alphas_cumprod=epsilon_net.alphas_cumprod.to(device),
        singular_value=diag.to(device),
        n_timesteps=len(epsilon_net.timesteps),
        var=std_obs**2,
        mode="else",
    )
    initial_noise = initial_noise.reshape(initial_noise.shape[0], -1)
    particles, weights = mcg_diff(
        initial_particles=initial_noise,
        observation=Ut_y,
        likelihood_diagonal=diag,
        score_model=score_model,
        coordinates_mask=coordinates_mask == 1,
        var_observation=(std_obs**2),
        timesteps=adapted_timesteps,
        eta=eta,
        n_samples_per_gpu_inference=64,
        gaussian_var=1e-4,
    )
    particles = particles[
        torch.randint(0, initial_noise.shape[0], size=(n_return,), device="cpu")
    ]
    return H_func.V(particles.to(initial_noise.device)).reshape(n_return, *dim_x)


def repaint(
    initial_noise_sample: torch.Tensor,
    timesteps: List[int],
    epsilon_net: EpsilonNet,
    obs: torch.Tensor,
    gibbs_steps: int,
    eta: float = 1.0,
):
    """for noiseless inpainting only"""
    sample = initial_noise_sample
    for i in tqdm.tqdm(range(len(timesteps) - 1, 1, -1)):
        t, t_prev = timesteps[i], timesteps[i - 1]
        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta
        )
        for g in range(gibbs_steps):
            alpha_cum_t_1, alpha_cum_t = (
                epsilon_net.alphas_cumprod[t_prev],
                epsilon_net.alphas_cumprod[t],
            )
            alpha_cum_t_1_to_t = alpha_cum_t / alpha_cum_t_1
            noised_obs = alpha_cum_t_1**0.5 * obs + (
                1 - alpha_cum_t_1
            ) ** 0.5 * torch.randn_like(obs)
            sample[:, : len(obs)] = noised_obs
            sample = alpha_cum_t_1_to_t**0.5 * sample + (
                1 - alpha_cum_t_1_to_t
            ) ** 0.5 * torch.randn_like(sample)
            sample = ddim_step(
                x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta
            )
    return epsilon_net.predict_x0(sample, timesteps[1])
