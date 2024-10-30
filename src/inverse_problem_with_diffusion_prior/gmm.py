import sys

import numpy as np
import numpyro.distributions as dist
import torch
from jax import grad, devices, default_device, value_and_grad, vmap
from jax import numpy as jnp
from jax.tree_util import Partial as partial
from jax.lax import fori_loop
from scipy.stats import wasserstein_distance
from torch import tensor, ones, eye, randn_like, randn, vstack, \
    manual_seed
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
from flows.rnvp import RealNVP
import optax
from ddrm.functions.denoising import efficient_generalized_steps_w_grad
from ddrm.functions.svd_replacement import GeneralH
from inverse_problem_with_diffusion_prior.ipwdp.generative_models import ScoreModel
from inverse_problem_with_diffusion_prior.ipwdp.inverse_problems_utils import NetReparametrized, build_extended_svd, gaussian_posterior,\
    get_optimal_timesteps_from_singular_values
from inverse_problem_with_diffusion_prior.ipwdp.nn_utils import EpsilonNetFromScore, Expandednet
from inverse_problem_with_diffusion_prior.ipwdp.optimal_particle_filter import particle_filter
import tqdm
from jax.random import normal, PRNGKey, split


def ou_mixt(alpha_t, means, dim, weights):
    cat = Categorical(weights)

    ou_norm = MultivariateNormal(
        vstack(tuple((alpha_t**.5) * m for m in means)),
        eye(dim).repeat(len(means), 1, 1))
    return MixtureSameFamily(cat, ou_norm)


def ou_mixt_jax(alpha_t, means, dim, weights):
    means = jnp.vstack(means)*(alpha_t**.5)
    covs = jnp.repeat(jnp.eye(dim)[None], axis=0, repeats=means.shape[0])
    return dist.MixtureSameFamily(component_distribution=dist.MultivariateNormal(means,
                                                                                 covariance_matrix=covs),
                                  mixing_distribution=dist.Categorical(weights))

def vi_training(key, logpdf, rnvp, optim, n_samples_step, n_train, n_final_samples):
    # pbar = trange(n_train, desc="Training NF")

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




def dps_jax(initial_particles,
            observation,
            A,
            target_dist_creator,
            timesteps,
            init_key,
            alphas_cumprod,
            zeta_coeff=9e-2,
            eta=1):
    x = initial_particles
    def lik(x, y, t):
        score_fn = grad(lambda x: target_dist_creator(alphas_cumprod[t]).log_prob(x).sum())
        score = score_fn(x)
        pred_x0 = (x + (1 - alphas_cumprod[t])*score) / (alphas_cumprod[t]**.5)
        pred = (A@pred_x0.T).T
        residue = pred - y[None, :]
        error = jnp.linalg.norm(residue, axis=-1)
        return (error**2).sum(), (pred_x0, error)

    grad_lik_fun = value_and_grad(lik, has_aux=True)
    keys = split(init_key, len(timesteps))
    for i, (t, t_prev) in tqdm.tqdm(enumerate(zip(timesteps[1:][::-1], timesteps[:-1][::-1])),
                                    desc='DPS'):
        (lik, (pred_x0, errors)), grad_lik = grad_lik_fun(x, observation, t)

        noise = eta * (((1 - alphas_cumprod[t_prev]) / (1 - alphas_cumprod[t])) * (
                1 - alphas_cumprod[t] / alphas_cumprod[t_prev])) ** .5
        coeff_sample = ((alphas_cumprod[t]/alphas_cumprod[t_prev])**.5)*((1 - alphas_cumprod[t_prev])/(1 - alphas_cumprod[t]))
        coeff_pred_x0 = ((alphas_cumprod[t_prev]**.5) * ((1 - alphas_cumprod[t] / alphas_cumprod[t_prev]))) / (1 - alphas_cumprod[t])

        gnoise = noise * normal(key=keys[i], shape=x.shape)
        nc_x = (coeff_sample * x + coeff_pred_x0*pred_x0 + gnoise)
        x = nc_x - grad_lik * (zeta_coeff/errors[:, None])
    return x


def sliced_wasserstein(dist_1, dist_2, n_slices=100):
    projections = torch.randn(size=(n_slices, dist_1.shape[1])).to(dist_1.device)
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = (projections @ dist_1.T)
    dist_2_projected = (projections @ dist_2.T)
    return np.mean([wasserstein_distance(u_values=d1.cpu().numpy(), v_values=d2.cpu().numpy()) for d1, d2 in zip(dist_1_projected, dist_2_projected)])


def get_posterior(obs, prior, A, Sigma_y):
    modified_means = []
    modified_covars = []
    weights = []
    precision = torch.linalg.inv(Sigma_y)
    for loc, cov, weight in zip(prior.component_distribution.loc,
                                prior.component_distribution.covariance_matrix,
                                prior.mixture_distribution.probs):
        new_dist = gaussian_posterior(obs,
                                      A,
                                      torch.zeros_like(obs),
                                      precision,
                                      loc,
                                      cov)
        modified_means.append(new_dist.loc)
        modified_covars.append(new_dist.covariance_matrix)
        prior_x = MultivariateNormal(loc=loc, covariance_matrix=cov)
        residue = obs - A @ new_dist.loc
        log_constant = -(residue[None, :] @ precision @ residue[:, None]) / 2 + \
                       prior_x.log_prob(new_dist.loc) - \
                       new_dist.log_prob(new_dist.loc)
        weights.append(torch.log(weight).item() + log_constant)
    weights = torch.tensor(weights)
    weights = weights - torch.logsumexp(weights, dim=0)
    cat = Categorical(logits=weights)
    ou_norm = MultivariateNormal(loc=torch.stack(modified_means, dim=0),
                                 covariance_matrix=torch.stack(modified_covars, dim=0))
    return MixtureSameFamily(cat, ou_norm)


def generate_measurement_equations(dim, dim_y, device, mixt):
    A = torch.randn((dim_y, dim))

    u, diag, coordinate_mask, v = build_extended_svd(A)
    diag = torch.sort(torch.rand_like(diag), descending=True).values

    A = u @ (torch.diag(diag) @ v[coordinate_mask == 1, :])
    init_sample = mixt.sample()
    std = (torch.rand((1,)))[0] * torch.ones(len(diag)) * max(diag)
    var_observations = std**2

    init_obs = A @ init_sample
    init_obs += randn_like(init_obs) * (var_observations**.5)
    Sigma_y = torch.diag(var_observations)
    posterior = get_posterior(init_obs, mixt, A, Sigma_y)
    return A, Sigma_y, u, diag, coordinate_mask, v, var_observations, posterior, init_obs


if __name__ == '__main__':
    use_gibbs = False
    n_samples = 2_000
    device = torch.device('cuda:0')
    n_particles_mcg_diff = 1_000
    T = 10
    delta = 0.01
    steps = int(T / delta)
    dists_infos = []
    save_folder = sys.argv[1]
    color_posterior = '#a2c4c9'
    color_algorithm = '#ff7878'
    for ind_increase, (n_steps, eta) in enumerate(zip([20, 100], [.6, .85])):
        for ind_dim, dim in enumerate([800, 80, 8]):
            # setup of the inverse problem
            means = []
            for i in range(-2, 3):
                means += [torch.tensor([-8.*i, -8.*j]*(dim//2)).to(device) for j in range(-2, 3)]
            weights = torch.randn(len(means))**2
            weights = weights / weights.sum()
            ou_mixt_fun = partial(ou_mixt,
                                  means=means,
                                  dim=dim,
                                  weights=weights)
            ou_mixt_jax_fun = partial(ou_mixt_jax,
                                      means=[jnp.array(m.numpy()) for m in means],
                                      dim=dim,
                                      weights=jnp.array(weights.numpy()))
            mixt = ou_mixt_fun(1)
            target_samples = mixt.sample((n_samples,)).cpu()
            for ind_ptg, dim_y in enumerate([1, 2, 4]):

                for i in range(20):
                    seed_num_inv_problem = (2**(ind_dim))*(3**(ind_ptg)*(5**(ind_increase))) + i
                    manual_seed(seed_num_inv_problem)
                    #dim_y = math.ceil(dim * ptg_y)
                    try:
                        A, Sigma_y, u, diag, coordinate_mask, v, var_observations, posterior, init_obs = generate_measurement_equations(dim, dim_y, device, mixt)
                    except ValueError:
                        seed_num_inv_problem += 1
                        manual_seed(seed_num_inv_problem)
                        A, Sigma_y, u, diag, coordinate_mask, v, var_observations, posterior, init_obs = generate_measurement_equations(dim,
                                                                                                                                        dim_y,
                                                                                                                                        device,
                                                                                                                                        mixt)

                    rnvp = RealNVP(n_features=dim, n_layer=10, n_hidden=128)

                    measure_jax = jnp.array(init_obs)
                    operator_jax = jnp.array(A)
                    sigma_y = var_observations[0].item()**.5
                    mixt_jax = ou_mixt_jax_fun(1)
                    def posterior_logprob(x):
                        return - ((measure_jax - operator_jax @ x) ** 2).sum(axis=-1) / (
                                2 * sigma_y ** 2) + mixt_jax.log_prob(x)


                    n_samples_step = 10
                    train_steps = 200
                    learning_rate = 1e-3

                    key, _ = split(PRNGKey(seed_num_inv_problem))
                    optim = optax.adam(learning_rate)
                    vi_train_func = partial(vi_training,
                                            rnvp=rnvp,
                                            logpdf=vmap(posterior_logprob),
                                            optim=optim,
                                            n_samples_step=n_samples_step,
                                            n_train=train_steps,
                                            n_final_samples=n_samples)
                    particles_vi, loss_vi = vi_train_func(key)
                    particles_vi = particles_vi.reshape(-1, dim)


                    betas = torch.linspace(.02, 1e-4, steps=999, device=device)
                    alphas_cumprod = torch.cumprod(tensor([1,] + [1 - beta for beta in betas]), dim=0) #they all ad alphas cumprod = 1 in the beginning
                    timesteps = torch.linspace(0, steps-1, n_steps, device=device).long()
                    adapted_timesteps = get_optimal_timesteps_from_singular_values(alphas_cumprod=alphas_cumprod,
                                                                                   singular_value=diag,
                                                                                   n_timesteps=n_steps,
                                                                                   var=var_observations[0].item(),
                                                                                   mode='else')

                    # score model
                    score_model = ScoreModel(NetReparametrized(base_score_module=EpsilonNetFromScore(ou_dist=ou_mixt_fun,
                                                                                                     alphas_cumprod=alphas_cumprod),
                                                               orthogonal_transformation=v),
                                             alphas_cumprod=alphas_cumprod,
                                             device=device)
                    score_model.net.device = device


                    # Getting posterior samples form nuts
                    posterior_samples = posterior.sample((n_samples,)).to(device)

                    # setting up parameters for ddrm and ours

                    n_particles = n_samples
                    initial_particles = randn(n_particles, dim).to(device)
                    # #dps
                    try:
                        with default_device(devices("gpu")[0]):
                            particles_dps = dps_jax(initial_particles=jnp.array(initial_particles.numpy()),
                                                    A=jnp.array(A.numpy()),
                                                    observation=jnp.array(init_obs.numpy()),
                                                    alphas_cumprod=jnp.array(alphas_cumprod.numpy()),
                                                    target_dist_creator=ou_mixt_jax_fun,
                                                    timesteps=adapted_timesteps.tolist(),
                                                    init_key=PRNGKey(seed_num_inv_problem),
                                                    zeta_coeff=1e-1,
                                                    eta=eta)
                            particles_dps = torch.from_numpy(np.array(particles_dps))
                    except Exception as e:
                        particles_dps = None
                    # # ddrm
                    H_funcs = GeneralH(H=A)
                    particles_ddrm = efficient_generalized_steps_w_grad(x=torch.randn(n_samples, 1, 1, dim).to(device),
                                                                        b=betas.to(device),
                                                                        seq=adapted_timesteps[:-1].tolist(),
                                                                        model=Expandednet(base_net=score_model.net.base_score_module,
                                                                                          expanded_size=(1, 1, dim)),
                                                                        y_0=init_obs[None, :].to(device),
                                                                        H_funcs=H_funcs,
                                                                        sigma_0=var_observations[0].item()**.5,
                                                                        etaB=1,
                                                                        etaA=.85,
                                                                        etaC=1,
                                                                        classes=None,
                                                                        cls_fn=None)[0][-1].flatten(1, 3).cpu()


                    particles_mcg_diff = []
                    n_batches_mcg_diff = n_samples // n_particles_mcg_diff
                    for batch_initial_particles in initial_particles.reshape(n_batches_mcg_diff,
                                                                             n_particles_mcg_diff,
                                                                             -1):
                        # MCGDIFF
                        particle_cloud = particle_filter(
                            initial_particles=batch_initial_particles,
                            observation=(u.T @ init_obs).to(device),
                            score_model=score_model,
                            likelihood_diagonal=diag,
                            coordinates_mask=coordinate_mask,
                            var_observation=var_observations[0].item(),
                            timesteps=adapted_timesteps,
                            eta=eta
                        )
                        particles_mcg_diff.append((v.T @ particle_cloud.T).T.cpu())
                    particles_mcg_diff = torch.cat(particles_mcg_diff, dim=0)
                    data = {
                        "seed": seed_num_inv_problem,
                        "sigma_y": sigma_y,
                        "D_X": dim,
                        "D_Y": dim_y,
                        "prior": mixt.sample((n_samples,)).cpu().numpy(),
                        "posterior": posterior_samples.cpu().numpy(),
                        "loss_RNVP": np.array(loss_vi),
                        "RNVP": np.array(particles_vi),
                        "DDRM": particles_ddrm.cpu().numpy(),
                        "DPS": np.array(particles_dps),
                        "MCG_DIFF": particles_mcg_diff.cpu().numpy()
                    }
                    np.savez(f'{save_folder}/{dim}_{dim_y}_{dim_y}_{25}_{seed_num_inv_problem}_{n_steps}.npz',
                             **data)