import tqdm

import torch
from torch import vmap
from torch.func import jacrev


class EpsNet(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps):
        super().__init__()
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps

    def forward(self, x, t):
        return self.net.predict_noise(x, torch.tensor(t))

    def predict_x0(self, x, t):
        alpha_cum_t = (
            self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        )
        return (x - (1 - alpha_cum_t) ** 0.5 * self.forward(x, t)) / (alpha_cum_t**0.5)

    def score(self, x, t):
        alpha_cum_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - alpha_cum_t) ** 0.5

    def value_and_grad_predx0(self, x, t):
        x = x.requires_grad_()
        pred_x0 = self.predict_x0(x, t)
        grad_pred_x0 = torch.autograd.grad(pred_x0.sum(), x)[0]
        return pred_x0, grad_pred_x0

    def value_and_jac_predx0(self, x, t):
        def pred(x):
            return self.predict_x0(x, t)

        pred_x0 = self.predict_x0(x, t)
        return pred_x0, vmap(jacrev(pred))(x)


class EpsilonNetSVD(EpsNet):
    def __init__(self, net, alphas_cumprod, timesteps, V):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.V = V
        self.timesteps = timesteps

    def forward(self, x, t):
        return (self.V.T @ self.net((self.V @ x.T).T, torch.tensor(t)).T).T


def bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsNet,
    ell: int,
    t: int,
    s: int,
    eta: float,
):
    """s < t < ell"""
    alpha_cum_s_to_t = epsilon_net.alphas_cumprod[t] / epsilon_net.alphas_cumprod[s]
    alpha_cum_t_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[t]
    alpha_cum_s_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[s]
    std = (
        eta
        * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
        ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)
    return coeff_xell * x_ell + coeff_xs * x_s, std


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsNet,
    ell: int,
    t: int,
    s: int,
    eta: float,
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return mean + std * torch.randn_like(mean)


def ddim_statistics(
    x: torch.Tensor, epsilon_net: EpsNet, t: float, t_prev: float, eta: float
):
    t_0 = epsilon_net.timesteps[0]
    return bridge_kernel_statistics(
        x_ell=x,
        x_s=epsilon_net.predict_x0(x, t),
        epsilon_net=epsilon_net,
        ell=t,
        t=t_prev,
        s=t_0,
        eta=eta,
    )


def ddim_step(
    x: torch.Tensor, epsilon_net: EpsNet, t: float, t_prev: float, eta: float
):
    t_0 = epsilon_net.timesteps[0]
    return sample_bridge_kernel(
        x_ell=x,
        x_s=epsilon_net.predict_x0(x, t),
        epsilon_net=epsilon_net,
        ell=t,
        t=t_prev,
        s=t_0,
        eta=eta,
    )


def ddim(
    initial_noise_sample: torch.Tensor, epsilon_net: EpsNet, eta: float = 1.0
) -> torch.Tensor:
    """
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    """
    sample = initial_noise_sample
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=epsilon_net.timesteps[i],
            t_prev=epsilon_net.timesteps[i - 1],
            eta=eta,
        )
    return epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])
