import sys
import math
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu/'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=30'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import matplotlib.pyplot as plt
from jax import numpy as jnp, value_and_grad, disable_jit
import numpy as np
from jax.random import normal, split, PRNGKey, PRNGKeyArray, uniform, randint
import numpyro
import optax
from jax.lax import fori_loop
import flax.linen as nn
from typing import Any, Callable
from jax.scipy.special import logsumexp


class PositionalEmbedding(nn.Module):

    num_channels: int
    max_positions: int = 1_000
    endpoint: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        freqs = jnp.arange(start=0, stop=self.num_channels//2, dtype=self.dtype)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions)**freqs
        x = jnp.outer(x, freqs)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)
        return x


class Net(nn.Module):
    def layer_once(self, y, dim):
        return nn.silu(nn.GroupNorm()(nn.Dense(dim)(y)))
    def unet(self, x, emb, out_dim):
        orig = x
        x = nn.Dense(out_dim)(x)
        skip = nn.Dense(out_dim)(orig)
        params = nn.Dense(features=out_dim)(emb)
        x = nn.GroupNorm()(x + params)
        x = nn.activation.silu(x)

        x = nn.Dense(out_dim)(x)
        x = x + skip
        return x
    @nn.compact
    def __call__(self, x, t):
        c_skip = 1 / (t**2 + 1)
        c_input = 1 / (t**2 + 1)**.5
        cout = t / (t**2 + 1)
        cnoise = 0.25 * jnp.log(t)
        x_emb = x*c_input[:, None]
        x_emb = nn.silu(nn.Dense(128)(x_emb))
        t_emb = PositionalEmbedding(num_channels=128,
                                    dtype=jnp.float32)(cnoise)
        t_emb = nn.activation.silu(nn.Dense(128)(t_emb))
        t_emb = nn.activation.silu(nn.Dense(128)(t_emb))

        for i in range(3):
            x_emb = self.unet(x_emb, t_emb, 256)
        for i in range(3):
            x_emb = self.unet(x_emb, t_emb, 512)
        for i in range(3):
            x_emb = self.unet(x_emb, t_emb, 256)
        for i in range(3):
            x_emb = self.unet(x_emb, t_emb, 128)
        return cout[:, None] * nn.Dense(x.shape[-1])(x_emb) + c_skip[:, None]*x


class FunnelJax(numpyro.distributions.Distribution):

    def __init__(self, a, b, loc, rot):
        super(FunnelJax).__init__()
        self.a = a
        self.b = b
        self._batch_shape = loc.shape[:-1]
        self._event_shape = loc.shape[-1:]
        self.loc = loc
        self.rotations = rot
        self.inv_rot = jnp.linalg.inv(rot)

    def sample(self, key, sample_shape):
        key_free_coords, _ = split(key)
        free_coords = normal(key_free_coords, shape=(*sample_shape, *self._batch_shape, *self._event_shape))
        stds = jnp.ones_like(free_coords)
        stds = stds.at[..., 0].multiply(self.a)
        stds = stds.at[..., 1:].multiply(jnp.exp(free_coords[..., 0] * self.b)[..., None])
        samples_funnel = free_coords * stds
        rotated_funnels = \
            (self.rotations.reshape(*((1,) * len(sample_shape)), *self.rotations.shape) @ samples_funnel[..., None])[
                ..., 0]
        return rotated_funnels + self.loc.reshape(*((1,) * len(sample_shape)), *self.loc.shape)

    def log_prob(self, value):
        value_shape = value.shape[:-2]
        unscaled = value - self.loc.reshape(*((1,) * len(value_shape)), *self.loc.shape)
        unrotated = (self.inv_rot.reshape(*((1,) * len(value_shape)), *self.inv_rot.shape) @ unscaled[..., None])[
            ..., 0]
        stds = jnp.ones_like(unrotated)
        stds = stds.at[..., 0].multiply(self.a)
        stds = stds.at[..., 1:].multiply(jnp.exp(unrotated[..., 0] * self.b)[..., None])
        return -jnp.linalg.norm(unrotated / stds, axis=-1) ** 2


def ve_kernel(x_t: jnp.ndarray,
              beta: float,
              key: PRNGKeyArray) -> jnp.ndarray:
    return x_t + normal(key=key, shape=x_t.shape)*(beta**.5)


def vp_kernel(x_t: jnp.ndarray,
              beta: float,
              key: PRNGKeyArray) -> jnp.ndarray:
    return ((1-beta)**.5) * x_t + (beta**.5)*normal(key=key ,shape=x_t.shape)


def bridge_vp(x_t: jnp.ndarray,
              x_0: jnp.ndarray,
              key: PRNGKeyArray,
              alphas_cumprod_t_1: float,
              alphas_cumprod_t: float) -> jnp.ndarray:
    bridge_var = ((1 - alphas_cumprod_t_1) / (1 - alphas_cumprod_t)) * (1 - alphas_cumprod_t / alphas_cumprod_t_1)
    mean = x_0 * (alphas_cumprod_t_1**.5)
    mean += (((1 - alphas_cumprod_t_1 - bridge_var)**.5) /((1 - alphas_cumprod_t)**.5))*(x_t - (alphas_cumprod_t**.5)*x_0)
    return mean + (bridge_var**.5)*normal(key=key, shape=x_t.shape)


def bridge_ve(x_t: jnp.ndarray,
              x_0: jnp.ndarray,
              key: PRNGKeyArray,
              std_t_1: float,
              std_t: float,
              beta_t: float) -> jnp.ndarray:
    bridge_var = ((std_t_1**2) * beta_t) / (std_t**2)
    mean = ((std_t_1**2)*x_t + beta_t*x_0) / (std_t**2)
    return mean + (bridge_var**.5)*normal(key=key, shape=x_t.shape)


def compute_kl_marginal_ve(samples: jnp.ndarray,
                           std: float,
                           key: PRNGKeyArray,
                           data_distr: numpyro.distributions.Distribution,
                           log_pdf_target: Callable[[jnp.ndarray,], jnp.ndarray]) -> jnp.ndarray:
    is_samples = data_distr.sample(key=key, sample_shape=(1, 1000,))
    kernel_log_density = - (jnp.linalg.norm(samples[..., None, :] - is_samples, axis=-1)**2) / (2 * (std**2)) - samples.shape[1]*jnp.log(std) - (samples.shape[1]/2)*jnp.log(2 * jnp.pi)
    marginal_log_pdf = logsumexp(kernel_log_density, axis=-1) - jnp.log(1000)
    return jnp.abs((marginal_log_pdf - log_pdf_target(samples)).mean(axis=0))


def compute_kl_marginal_vp(samples: jnp.ndarray,
                           alpha_cumprods: float,
                           key: PRNGKeyArray,
                           data_distr: numpyro.distributions.Distribution,
                           log_pdf_target: Callable[[jnp.ndarray,], jnp.ndarray]) -> jnp.ndarray:
    is_samples = data_distr.sample(key=key, sample_shape=(1, 1000,))
    means = (alpha_cumprods ** .5) * is_samples
    var = (1 - alpha_cumprods)
    kernel_log_density = - (jnp.linalg.norm(samples[..., None, :] - means, axis=-1)**2) / (2 * var) - (samples.shape[-1]/2)*jnp.log(var) - (samples.shape[-1]/2)*jnp.log(2 * jnp.pi)
    marginal_log_pdf = logsumexp(kernel_log_density, axis=-1) - jnp.log(1000)
    return (marginal_log_pdf - log_pdf_target(samples)).mean(axis=0)


def train_ve_score(model: nn.Module,
                   data_dist: numpyro.distributions.Distribution,
                   n_epochs: int,
                   batch_size: int,
                   key: PRNGKeyArray,
                   stds: jnp.ndarray,
                   lr: float = 1e-3):
    optim = optax.adam(lr)
    keys = split(key, num=n_epochs*3).reshape(n_epochs, 3, -1)

    def train_step(i, state):
        params, opt_state, losses = state
        data_key, noise_key, noise_level_key = keys[i]
        batch_data = data_dist.sample(key=data_key, sample_shape=(batch_size,))
        sampled_stds = jnp.exp(normal(key=noise_level_key, shape=(batch_size,))*1.2 - 1.2).clip(stds[0], stds[-1])
        corrupt_data = batch_data + sampled_stds[:, None] * normal(key=noise_key, shape=batch_data.shape)
        weights = 1 / (sampled_stds[:, None]**2)
        def loss_fn(params):
            predicted_data = model.apply({'params': params}, corrupt_data, sampled_stds)
            return (jnp.linalg.norm(predicted_data - batch_data, axis=-1)**2).sum(), predicted_data

        (loss, pred), grads = value_and_grad(loss_fn, has_aux=True)(params)
        losses = losses.at[i].set(loss)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, losses)

    losses = jnp.empty(n_epochs)

    init_params = model.init(key, jnp.ones((1, 2)), jnp.ones((1,)))["params"]
    init_opt_state = optim.init(init_params)

    param, _, losses = fori_loop(
        body_fun=train_step,
        lower=0,
        upper=n_epochs,
        init_val=(init_params, init_opt_state, losses)
    )

    return param, losses


def train_vp_score(model: nn.Module,
                   data_dist: numpyro.distributions.Distribution,
                   n_epochs: int,
                   batch_size: int,
                   key: PRNGKeyArray,
                   alphas_cumprod: jnp.ndarray,
                   lr: float = 1e-3):
    optim = optax.adam(lr)
    keys = split(key, num=n_epochs*3).reshape(n_epochs, 3, -1)

    def train_step(i, state):
        params, opt_state, losses = state
        data_key, noise_key, noise_level_key = keys[i]
        batch_data = data_dist.sample(key=data_key, sample_shape=(batch_size,))
        noise_level = randint(key=noise_level_key, minval=0, maxval=len(alphas_cumprod), shape=(batch_size,))
        afs = alphas_cumprod[noise_level]
        corrupt_data = batch_data*(afs**.5)[:, None] + ((1 - afs)**.5)[:, None] * normal(key=noise_key, shape=batch_data.shape)
        def loss_fn(params):
            predicted_data = model.apply({'params': params}, corrupt_data, noise_level)
            return (jnp.linalg.norm(predicted_data - batch_data, axis=-1)**2).mean()

        loss, grads = value_and_grad(loss_fn)(params)
        losses = losses.at[i].set(loss)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, losses)

    losses = jnp.empty(n_epochs)

    init_params = model.init(key, jnp.ones((1, 2)), jnp.ones((1,)))["params"]
    init_opt_state = optim.init(init_params)

    param, _, losses = fori_loop(
        body_fun=train_step,
        lower=0,
        upper=n_epochs,
        init_val=(init_params, init_opt_state, losses)
    )

    return param, losses


if __name__ == '__main__':
    seed = 0
    color_posterior = '#00428d'
    color_target = '#fa526c'
    plt.rcParams.update({'font.size': 22})
    locs = uniform(key=PRNGKey(seed), shape=(10, 2))*30 - 15
    rotations = normal(key=PRNGKey(seed+1), shape=(10, 2, 2))
    U, _, VT = jnp.linalg.svd(rotations)
    rotations = U @ VT
    probs = uniform(key=PRNGKey(seed+2), shape=(10,))
    probs = probs / probs.sum()

    cat = numpyro.distributions.Categorical(probs)
    fun = FunnelJax(a=1., b=.5, loc=locs, rot=rotations)
    fun_mixture = numpyro.distributions.MixtureSameFamily(cat, fun)

    betas_vp = jnp.linspace(1e-4, 0.02, 1000)
    alphas_cumprod_vp = jnp.cumprod(1 - betas_vp)

    T = 1000
    sigma_max = 25
    sigma_min = .005
    p = 5
    std_ve = jnp.arange(0, T - 1) / (T - 2)
    std_ve = ((sigma_max ** (1 / p) + std_ve * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** (p))[::-1]
    betas_ve = jnp.diff(std_ve**2)
    print(std_ve[-1])
    target_dist_vp = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(2), covariance_matrix=jnp.eye(2))
    target_dist_ve = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(2), covariance_matrix=(std_ve[-1]**2)*jnp.eye(2))

    initial_samples = fun_mixture.sample(key=PRNGKey(seed+3), sample_shape=(1000,))
    samples_ve = jnp.copy(initial_samples)
    samples_vp = jnp.copy(samples_ve)

    c_fwd = 'blue'
    c_initial = 'red'
    img_counter = 0
    for i in range(1000):
        samples_ve = ve_kernel(samples_ve, beta=betas_ve[i], key=PRNGKey(seed + 4 + i))
        samples_vp = vp_kernel(samples_vp, beta=betas_vp[i], key=PRNGKey(2*(seed + 4 + i)))
        if i%10 == 9:
            kl_ve = compute_kl_marginal_ve(samples_ve,
                                           std_ve[i],
                                           PRNGKey(3*(seed + 4 + i)),
                                           data_distr=fun_mixture,
                                           log_pdf_target=target_dist_ve.log_prob
                                           )
            kl_vp = compute_kl_marginal_vp(
                samples_vp,
                alphas_cumprod_vp[i],
                PRNGKey(4 * (seed + 4 + i)),
                data_distr=fun_mixture,
                log_pdf_target=target_dist_vp.log_prob
            )
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.subplots_adjust(left=0, right=1, wspace=0, bottom=0, top=.9)
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim([-20, 20])
                ax.set_xlim([-20, 20])
            axes[0].scatter(*samples_ve.T, color=color_target, alpha=.3, rasterized=True)
            axes[0].scatter(*initial_samples.T, color=color_posterior, alpha=.1, rasterized=True)
            axes[1].scatter(*samples_vp.T, color=color_target, alpha=.3, rasterized=True)
            axes[1].scatter(*initial_samples.T, color=color_posterior, alpha=.1, rasterized=True)
            axes[0].set_title(f'KL = {kl_ve:.2f}')
            axes[1].set_title(f'KL = {kl_vp:.2f}')
            fig.savefig(f'images/animation_forward/images_{img_counter}.pdf')
            img_counter += 1
            plt.close(fig)

    model_ve = Net()
    with disable_jit(False):
        params_ve, loss_ve = train_ve_score(model_ve,
                                            fun_mixture,
                                            100_000,
                                            32,
                                            PRNGKey(seed + 5),
                                            stds=std_ve,
                                            lr=1e-6)

    params_vp, loss_vp = train_vp_score(model_ve,
                                        fun_mixture,
                                        1_000,
                                        32,
                                        PRNGKey(seed + 5),
                                        alphas_cumprod=alphas_cumprod_vp,
                                        lr=1e-3)

    plt.plot(loss_vp)
    plt.plot(loss_ve)
    plt.yscale('log')
    plt.show()

    backward_kernel_vp = lambda x_t, t, t_prev, key: bridge_vp(
        x_t=x_t,
        x_0=model_ve.apply({"params": params_vp},
                           x_t, t),
        alphas_cumprod_t_1=alphas_cumprod_vp[t_prev],
        alphas_cumprod_t=alphas_cumprod_vp[t],
        key=key
    )
    backward_kernel_ve = lambda x_t, t, t_prev, key: bridge_ve(
        x_t=x_t,
        x_0=model_ve.apply({"params": params_ve},
                           x_t, std_ve[t, None]),
        std_t_1=std_ve[t_prev, None],
        std_t=std_ve[t, None],
        beta_t=betas_ve[t, None],
        key=key
    )

    samples_backward_vp = target_dist_vp.sample(key=PRNGKey(seed + 6),
                                                sample_shape=(1000,))
    samples_backward_ve = target_dist_ve.sample(key=PRNGKey(seed + 7),
                                                sample_shape=(1000,))
    t_linspace = range(1000, -50, -50)
    for k, t, t_prev in zip(split(PRNGKey(seed + 8), len(t_linspace) - 1),
                            t_linspace[:-1],
                            t_linspace[1:]):
        samples_backward_vp = backward_kernel_vp(samples_backward_vp, t, t_prev, k)
        samples_backward_ve = backward_kernel_ve(samples_backward_ve, t, t_prev, k)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.subplots_adjust(left=0, right=1, wspace=0, bottom=0, top=.9)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([-20, 20])
            ax.set_xlim([-20, 20])
        axes[0].scatter(*samples_backward_ve.T, color=c_fwd, alpha=.3)
        axes[0].scatter(*initial_samples.T, color=c_initial, alpha=.1)
        axes[1].scatter(*samples_backward_vp.T, color=c_fwd, alpha=.3)
        axes[1].scatter(*initial_samples.T, color=c_initial, alpha=.1)
        # axes[0].set_title(f'KL = {kl_ve:.2f}')
        # axes[1].set_title(f'KL = {kl_vp:.2f}')

        fig.show()
        plt.close(fig)










