import jax.numpy as jnp
import optax
import equinox as eqx
from jax.lax import fori_loop
from flowMC.nfmodel.utils import make_training_loop
from numpyro.distributions import MultivariateNormal
from rnvp import RealNVP
from jax.random import PRNGKey, split, multivariate_normal

from jax.tree_util import Partial as partial
from tqdm import trange
import os
from jax import default_device, devices, jit, vmap, debug


def vi_training(key, logpdf, rnvp, optim, n_samples, n_train):
    pbar = trange(n_train, desc="Training NF")

    def forward_kl(params, noise):
        samples, logdet = model.apply({'params': params}, rnvp.inverse)
        return (- (noise ** 2).sum(-1) / 2 - logdet - logpdf(samples)).mean()

    @eqx.filter_jit
    def train_step(i, params, keys):
        noise = multivariate_normal(jnp.zeros(dim), jnp.eye(dim)).sample(keys[i], n_samples)
        loss, grads = forward_kl(rnvp, noise)
        updates, opt_state = optim.update(grads, state)
        rnvp = eqx.apply_updates(rnvp, updates)
        return (opt_state)

    keys = split(key, n_train)
    # train_func = partial(train_step, keys=keys)
    state = optim.init(eqx.filter(rnvp, eqx.is_array))
    vi_step_func = partial(train_step, keys=keys)
    fori_loop(0, n_train, vi_step_func, (rnvp, state))
    for i in pbar:
        kl_val, model, state = train_func(i, model, state)
        if i % 10 == 0:
            pbar.set_description(f"forward KL current val: {kl_val}")
    return model

