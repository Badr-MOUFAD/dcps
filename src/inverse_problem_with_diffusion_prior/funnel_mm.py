import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu/'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=30'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import collections
import functools
import math
import random
#
import torch
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from inverse_problem_with_diffusion_prior.ipwdp.generative_models import ddim_sampling
from inverse_problem_with_diffusion_prior.ipwdp.generative_models import ScoreModel
from inverse_problem_with_diffusion_prior.ipwdp.optimal_particle_filter import particle_filter
from inverse_problem_with_diffusion_prior.ipwdp.inverse_problems_utils import get_posterior_distribution_from_dist

from flows.rnvp import RealNVP
# import pyro.infer.mcmc as mcmc
from inverse_problem_with_diffusion_prior.ipwdp.inverse_problems_utils import NetReparametrized, sliced_wasserstein
from dps.guided_diffusion.gaussian_diffusion import get_sampler, space_timesteps
from dps.guided_diffusion.condition_methods import PosteriorSampling
from dps.guided_diffusion.measurements import LinearOperator, get_noise
from ddrm.functions.denoising import efficient_generalized_steps
from inverse_problem_with_diffusion_prior.ipwdp.svd_replacement import GeneralH
from inverse_problem_with_diffusion_prior.ipwdp.nn_utils import EpsilonNetFromScore, Expandednet
import optax
#
import blackjax
from jax import default_device, devices
from jax.random import normal, split, PRNGKey
from jax.lax import fori_loop
import numpyro  # from numpyro.distributions import Distribution, Normal, Categorical, MixtureSameFamily
from jax import vmap, numpy as jnp, jit, pmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import trange


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


def bkjx_loop(key, init_state, kernel, steps):
    keys = split(key, steps)

    def one_step(i, state):
        state, _ = kernel.step(keys[i], state)
        return state

    return fori_loop(0, steps, one_step, kernel.init(init_state))


def nuts_warmup(key, sample, logprob_fun):
    key_warmup, key_nuts = split(key, 2)
    res = blackjax.window_adaptation(blackjax.nuts, logprob_fun).run(key_warmup, {"loc": sample})
    inverse_mass_matrix = res[-1][2].mm_state.inverse_mass_matrix[-1]
    step_size = jnp.exp(res[-1][2].da_state.log_step_size_avg[-1])
    state = res[-1][0].position['loc'][-1]
    return state, step_size, inverse_mass_matrix


def nuts_once(key, state, step_size, inverse_mass_matrix, logprob_fun, num_steps):
    nuts = blackjax.nuts(logprob_fun, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size)
    nuts_sample = bkjx_loop(key, kernel=nuts, init_state={"loc": state},
                            steps=num_steps).position['loc']
    return nuts_sample


def sample_nuts(D_X, key, mixt_dist_jax, n_chains, measure_jax, operator_jax, sigma_y):
    def posterior_logprob_nuts(x):
        return - ((measure_jax - operator_jax @ x['loc']) ** 2).sum(axis=-1) / (
                2 * sigma_y ** 2) + mixt_dist_jax.log_prob(x['loc'])

    posterior_logprob_nuts({'loc': jnp.zeros(D_X)})
    key_warmup, key_nuts, key_categorical = split(key, 3)
    batch_size = len(devices("cpu"))
    n_batches = math.ceil(n_chains / batch_size)
    nuts_warmup_fun = pmap(partial(nuts_warmup, logprob_fun=posterior_logprob_nuts),
                           devices=devices("cpu"))
    nuts_warmups = [nuts_warmup_fun(
        split(k, batch_size),
        mixt_dist_jax.sample(k,
                             sample_shape=(batch_size,))) for k in
        tqdm.tqdm(split(key_warmup, n_batches), desc='NUTS Warmup')]
    initial_positions = jnp.concatenate([x[0] for x in nuts_warmups])
    step_sizes = jnp.concatenate([x[1] for x in nuts_warmups])
    inverse_mass_matrixes = jnp.concatenate([x[2] for x in nuts_warmups])
    logits = vmap(posterior_logprob_nuts)({"loc": initial_positions})
    ancestors = numpyro.distributions.Categorical(logits=logits).sample(key_categorical,
                                                                        sample_shape=(logits.shape[0],))
    initial_positions = initial_positions[ancestors]
    step_sizes = step_sizes[ancestors]
    inverse_mass_matrixes = inverse_mass_matrixes[ancestors]
    nuts_sampler = pmap(partial(nuts_once, logprob_fun=posterior_logprob_nuts, num_steps=10_000),
                        devices=devices("cpu"))
    nuts_samples = jnp.concatenate([nuts_sampler(
        split(k, batch_size),
        init,
        step,
        inv_mass)
        for k, init, step, inv_mass in tqdm.tqdm(zip(split(key, n_batches),
                                                     initial_positions.reshape(-1, batch_size,
                                                                               initial_positions.shape[-1]),
                                                     step_sizes.reshape(-1, batch_size),
                                                     inverse_mass_matrixes.reshape(-1, batch_size,
                                                                                   *inverse_mass_matrixes.shape[1:])),
                                                 desc='NUTS')])
    return nuts_samples


def vi_training(key, logpdf, rnvp, optim, n_samples_step, n_train, n_final_samples):

    def train_step(i, state, keys):
        params, opt_state, losses = state
        noise = normal(keys[i], (n_samples_step, rnvp.n_features))

        def forward_kl(params):
            samples, logdet = rnvp.apply({'params': params}, noise, method=rnvp.inverse)
            return (- (noise ** 2).sum(-1) / 2 - logdet - logpdf(samples)).mean()

        loss, grads = value_and_grad(forward_kl)(params)
        losses = losses.at[i].set(loss)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, losses)

    key_init, key_train, key_samples = split(key, 3)
    keys = split(key_train, n_train)
    losses = jnp.empty(n_train)

    init_params = rnvp.init(key_init, jnp.ones((1, rnvp.n_features)))["params"]
    init_opt_state = optim.init(init_params)
    vi_step_func = partial(train_step, keys=keys)
    params, _, losses = fori_loop(0, n_train, vi_step_func, (init_params, init_opt_state, losses))
    return rnvp.apply({'params': params}, key_samples, n_final_samples, method=rnvp.sample), losses


class LOperator(LinearOperator):
    def __init__(self, A, **kwargs):
        super().__init__()
        self.A = A

    def forward(self, data, **kwargs):
        return (self.A @ data.T).T

    def transpose(self, data, **kwargs):
        return (self.A.T @ data.T).T


class Funnel(torch.distributions.Distribution):

    def __init__(self, a, b, loc, rot):
        super(Funnel).__init__()
        self.a = a
        self.b = b
        self.free_noise_dist = torch.distributions.Normal(0, 1)
        self._batch_shape = loc.shape[:-1]
        self._event_shape = loc.shape[-1:]
        self.loc = loc
        self.rotations = rot
        self.inv_rot = torch.linalg.inv(rot)

    def sample(self, sample_shape):
        free_coords = self.free_noise_dist.sample(
            sample_shape=(*sample_shape, *self._batch_shape, *self._event_shape)).to(self.loc.device)
        stds = torch.ones_like(free_coords)
        stds[..., 0] *= self.a
        stds[..., 1:] *= torch.exp(free_coords[..., 0] * self.b)[..., None]
        samples_funnel = free_coords * stds
        rotated_funnels = \
            (self.rotations.reshape(*((1,) * len(sample_shape)), *self.rotations.shape) @ samples_funnel[..., None])[
                ..., 0]
        return rotated_funnels + self.loc.reshape(*((1,) * len(sample_shape)), *self.loc.shape)

    def log_prob(self, value):
        value_shape = value.shape[:-2]
        unscalled = value - self.loc.reshape(*((1,) * len(value_shape)), *self.loc.shape)
        unrottated = (self.inv_rot.reshape(*((1,) * len(value_shape)), *self.inv_rot.shape) @ unscalled[..., None])[
            ..., 0]
        stds = torch.ones_like(unrottated).to(self.loc.device)
        stds[..., 0] *= self.a
        stds[..., 1:] *= torch.exp(unrottated[..., 0] * self.b)[..., None]
        return self.free_noise_dist.log_prob(unrottated / stds).sum(-1)


def learn_score(
        mixt_dist: torch.distributions.MixtureSameFamily,
        alphas_cumprod: torch.Tensor,
        net: torch.nn.Module,
        tol: float,
        lr: float = 1e-3,
        batch_size: int = 256,
        n_max: int = 1_000_000,
        val_period: int = 100,
):
    optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
    pbar = tqdm.tqdm(range(n_max),
                     desc='Score Training')

    loss_buffer = collections.deque([], maxlen=1)
    for i in pbar:
        samples = mixt_dist.sample((batch_size,))
        noises = torch.randn_like(samples)
        times = torch.randint(low=0, high=len(alphas_cumprod), size=(batch_size, 1))
        alphas = alphas_cumprod[times[:, 0]]
        perturbed_version = samples * (alphas[..., None] ** .5) + ((1 - alphas[..., None]) ** .5) * noises

        optimizer.zero_grad()
        predicted_noise = net(perturbed_version,
                              times)
        loss = torch.nn.functional.mse_loss(input=predicted_noise,
                                            target=noises)
        loss.backward()
        optimizer.step()
        if i % val_period == (val_period - 1):
            score_model = ScoreModel(net=net.eval(),
                                     alphas_cumprod=alphas_cumprod,
                                     device='cuda')
            with torch.no_grad():
                samples = ddim_sampling(initial_noise_sample=torch.randn_like(samples),
                                        timesteps=torch.arange(0, len(alphas_cumprod)).tolist()[::-1],
                                        score_model=score_model,
                                        eta=1)
            lw = mixt_dist.log_prob(samples).mean()
            lw_normal = mixt_dist.log_prob(mixt_dist.sample((samples.shape[0],))).mean()
            pbar.set_postfix({'LW ptg': (lw / lw_normal).item()}, refresh=False)
            loss_buffer.append(lw.item())
            if lw / lw_normal > tol:
                break
    return net


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[x.flatten()]


class Net(torch.nn.Module):

    def     __init__(self, dim_input, dim_embedding=512, n_layers=3):
        super().__init__()
        self.input_layer = torch.nn.Sequential(torch.nn.Linear(dim_input, dim_embedding),
                                               torch.nn.ReLU())
        def res_layer_maker(dim_in, dim_out):
            return torch.nn.Sequential(torch.nn.Linear(dim_in, 2*dim_out),
                                       torch.nn.BatchNorm1d(2*dim_out),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * dim_out, 2 * dim_out),
                                       torch.nn.BatchNorm1d(2 * dim_out),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * dim_out, dim_out),
                                       torch.nn.BatchNorm1d(dim_out),
                                       torch.nn.ReLU(),
                                       )
        self.res_layers = torch.nn.ModuleList([res_layer_maker(dim_embedding, dim_embedding) for i in range(n_layers)])
        self.final_layer = torch.nn.Linear(dim_embedding, dim_input)
        self.time_embedding = PositionalEncoding(d_model=dim_embedding)

    def forward(self, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=x.device)
        x_emb = self.input_layer(x)
        t_emb = self.time_embedding(t.long())
        for lr in self.res_layers:
            x_emb = lr(t_emb + x_emb) + x_emb
        return self.final_layer(x_emb) - x


def generate_inverse_problem(D_X, N_components, box_size=20, operator_rank=1, dim_y=2, sigma_y=.1):
    locs = torch.rand(size=(N_components, D_X)) * box_size - box_size // 2
    rotations = torch.randn(size=(N_components, D_X, D_X))
    U, _, VT = torch.linalg.svd(rotations)
    rotations = torch.bmm(U, VT)
    probs = torch.rand(N_components, device='cuda')
    probs = probs / probs.sum()
    mixt_dist = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=Funnel(a=1., b=.5, loc=locs.cuda(),
                                      rot=rotations.cuda()),
        validate_args=False)

    cat = numpyro.distributions.Categorical(jnp.array(probs.cpu()))
    fun = FunnelJax(a=1., b=.5, loc=jnp.array(locs.cpu()), rot=jnp.array(rotations.cpu()))
    fun_mixture = numpyro.distributions.MixtureSameFamily(cat, fun)

    U, diag, VT = torch.linalg.svd(torch.randn(size=(dim_y, D_X), device=mixt_dist.component_distribution.loc.device),
                                   full_matrices=True)
    diag[operator_rank:] = 0

    operator = U @ torch.cat((torch.diag(diag), torch.zeros((dim_y, D_X - dim_y), device=diag.device)), axis=1) @ VT
    x_origin = mixt_dist.sample(sample_shape=(1,))[0]
    measure = operator @ x_origin
    measure = measure + torch.randn_like(measure) * sigma_y

    return mixt_dist, fun_mixture, measure, operator, x_origin, U, diag, VT


def get_data(D_X, D_Y, operator_rank, N_components, seed, n_samples=10_000,
             mcg_diff_n_particles=1_000,
             dps_batch_size=None,
             ddrm_batch_size=None):
    torch.manual_seed(seed)
    key = PRNGKey(seed)
    sigma_y = torch.rand(size=(1,)).item()
    mixt_dist, mixt_dist_jax, measure, operator, x_origin, U, diag, VT = generate_inverse_problem(D_X,
                                                                                                  N_components,
                                                                                                  box_size=30,
                                                                                                  sigma_y=sigma_y,
                                                                                                  operator_rank=operator_rank,
                                                                                                  dim_y=D_Y)
    n_chains = n_samples
    operator_jax = jnp.array(operator.cpu())
    measure_jax = jnp.array(measure.cpu())

    nuts_samples = sample_nuts(D_X, key, mixt_dist_jax, n_chains, operator_jax=operator_jax, measure_jax=measure_jax, sigma_y=sigma_y)

    rnvp = RealNVP(n_features=D_X, n_layer=10, n_hidden=128)
    def posterior_logprob(x):
        return - ((measure_jax - operator_jax @ x) ** 2).sum(axis=-1) / (
                2 * sigma_y ** 2) + mixt_dist_jax.log_prob(x)

    n_samples_step = 10
    train_steps = 200
    learning_rate = 1e-3

    key, _ = split(key)
    optim = optax.adam(learning_rate)
    vi_train_func = partial(vi_training,
                            rnvp=rnvp,
                            logpdf=vmap(posterior_logprob),
                            optim=optim,
                            n_samples_step=n_samples_step,
                            n_train=train_steps,
                            n_final_samples=n_samples)
    particles_vi, loss_vi = vi_train_func(key)
    particles_vi = particles_vi.reshape(-1, D_X)

    net = Net(dim_input=D_X,
              dim_embedding=512, n_layers=5).to('cuda')
    betas = torch.linspace(1e-4, 0.01, steps=1000)
    alphas_cumprod = torch.cumprod(1 - betas, 0).cuda()
    net = learn_score(mixt_dist=mixt_dist,
                      alphas_cumprod=alphas_cumprod,
                      net=net,
                      lr=1e-3,
                      batch_size=512,
                      tol=10,
                      n_max=10000,
                      val_period=1000)
    score_model = ScoreModel(net=net.requires_grad_(False).eval(),
                             alphas_cumprod=alphas_cumprod,
                             device='cuda')
    with torch.no_grad():
        samples = ddim_sampling(initial_noise_sample=torch.randn(n_samples, D_X).cuda(),
                                timesteps=torch.arange(0, len(alphas_cumprod)).tolist()[::-1],
                                score_model=score_model,
                                eta=1).cpu()


    # mcg_diff
    total_N = n_samples
    n_particles = mcg_diff_n_particles
    N_batches = math.ceil(total_N / n_particles)
    initial_particles = torch.randn(size=(N_batches, n_particles, D_X))
    particles_mcg_diff = []
    U_t_y_0 = U.T @ measure
    coordinates_mask = torch.cat((torch.ones(operator_rank), torch.zeros(D_X - operator_rank)), dim=-1)
    timesteps = torch.arange(0, len(alphas_cumprod), 10)
    eta = 1
    for j, batch_initial_particles in enumerate(initial_particles):

        particles = particle_filter(initial_particles=batch_initial_particles.cuda(),
                                    observation=U_t_y_0.cuda(),
                                    likelihood_diagonal=diag.cuda(),
                                    score_model=ScoreModel(net=NetReparametrized(base_score_module=score_model.net,
                                                                                 orthogonal_transformation=VT),
                                                           alphas_cumprod=alphas_cumprod,
                                                           device='cuda'),
                                    coordinates_mask=coordinates_mask.cuda(),
                                    var_observation=sigma_y ** 2,
                                    timesteps=timesteps.cuda(),
                                    eta=eta,
                                    n_samples_per_gpu_inference=n_particles,
                                    gaussian_var=1e-4)
        particles = (VT.T @ particles.T).T
        particles_mcg_diff.append(particles.cpu())
    particles_mcg_diff = torch.concat(particles_mcg_diff, dim=0)

    ### DPS
    batch_size = dps_batch_size if dps_batch_size else n_samples
    N_batches = n_samples // batch_size
    initial_particles = torch.randn(size=(N_batches, batch_size, D_X)).cpu()
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    sampler = get_sampler(name='ddim')(
        use_timesteps=space_timesteps(len(alphas_cumprod), f'ddim{len(timesteps)}'),
        betas=betas.cpu(),
        model_mean_type='epsilon',
        model_var_type='fixed_small',
        dynamic_threshold=False,
        clip_denoised=False,
        rescale_timesteps=False
    )
    particles_dps = []
    for j, batch_initial_particles in enumerate(initial_particles):
        particles_dps.append(sampler.p_sample_loop(model=net,
                                                   x_start=batch_initial_particles.cuda(),
                                                   measurement=measure[None, ...].repeat(
                                                       batch_initial_particles.shape[0],
                                                       1),
                                                   measurement_cond_fn=PosteriorSampling(
                                                       operator=LOperator(A=operator),
                                                       noiser=get_noise('gaussian', sigma=sigma_y),
                                                       scale=3e-1).conditioning,
                                                   record=False,
                                                   save_root=False).cpu())
    particles_dps = torch.concatenate(particles_dps, dim=0)

    ### DDRM
    batch_size = ddrm_batch_size if ddrm_batch_size else n_samples
    N_batches = n_samples // batch_size
    ddrm_timesteps = timesteps.clone()
    ddrm_timesteps[-1] = ddrm_timesteps[-1] - 1
    initial_particles = torch.randn((N_batches, batch_size, D_X)).cpu()
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    particles_ddrm = []
    for j, batch_initial_particles in enumerate(initial_particles):
        particles_ddrm.append(efficient_generalized_steps(
            x=batch_initial_particles[:, None, None, :].cuda(),
            b=betas,
            seq=ddrm_timesteps.cpu(),
            model=Expandednet(base_net=net, expanded_size=(1, 1, D_X)),
            y_0=measure[None, None, None, :].cuda(),
            H_funcs=GeneralH(H=operator),
            sigma_0=sigma_y,
            etaB=.85,
            etaA=1,
            etaC=1,
            classes=None,
            cls_fn=None)[0][-1].cpu())
    particles_ddrm = torch.concat(particles_ddrm, dim=0)[:, 0, 0]
    return {
        "seed": seed,
        "sigma_y": sigma_y,
        "D_X": D_X,
        "D_Y": D_Y,
        "prior": mixt_dist.sample((n_samples,)).cpu().numpy(),
        "prior_diffusion": samples.cpu().numpy(),
        "loss_RNVP": np.array(loss_vi),
        "RNVP": particles_vi,
        "DDRM": particles_ddrm,
        "DPS": particles_dps,
        "NUTS": nuts_samples,
        "MCG_DIFF": particles_mcg_diff
    }


if __name__ == '__main__':
    folder = sys.argv[1]
    for D_X in range(6, 8, 2):
        for D_Y in range(1, D_X, 2):
            for N_components in range(20, 25, 5):
                for seed in range(20):
                    data = get_data(D_X, D_Y, D_Y, N_components, seed)
                    np.savez(f'{folder}/{D_X}_{D_Y}_{D_Y}_{N_components}_{seed}.npz',
                             **data)


