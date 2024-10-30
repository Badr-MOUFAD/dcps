import torch
from functools import partial
from mcg_diff.particle_filter import mcg_diff
from mcg_diff.sgm import ScoreModel
from mcg_diff.utils import NetReparametrized, get_optimal_timesteps_from_singular_values

def ou_mixt(alpha_t, means, dim, weights):
    cat = torch.distributions.Categorical(weights, validate_args=False)

    ou_norm = torch.distributions.MultivariateNormal(
        torch.vstack(tuple((alpha_t**.5) * m for m in means)),
        torch.eye(dim).repeat(len(means), 1, 1), validate_args=False)
    return torch.distributions.MixtureSameFamily(cat, ou_norm, validate_args=False)


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
        prior_x = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
        residue = obs - A @ new_dist.loc
        log_constant = -(residue[None, :] @ precision @ residue[:, None]) / 2 + \
                       prior_x.log_prob(new_dist.loc) - \
                       new_dist.log_prob(new_dist.loc)
        weights.append(torch.log(weight).item() + log_constant)
    weights = torch.tensor(weights)
    weights = weights - torch.logsumexp(weights, dim=0)
    cat = torch.distributions.Categorical(logits=weights)
    ou_norm = torch.distributions.MultivariateNormal(loc=torch.stack(modified_means, dim=0),
                                                     covariance_matrix=torch.stack(modified_covars, dim=0))
    return torch.distributions.MixtureSameFamily(cat, ou_norm)


def gaussian_posterior(y,
                       likelihood_A,
                       likelihood_bias,
                       likelihood_precision,
                       prior_loc,
                       prior_covar):
    prior_precision_matrix = torch.linalg.inv(prior_covar)
    posterior_precision_matrix = prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
    posterior_mean = posterior_covariance_matrix @ (likelihood_A.T @ likelihood_precision @ (y - likelihood_bias) + prior_precision_matrix @ prior_loc)
    try:
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return torch.distributions.MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix, validate_args=False)
    except ValueError:
        u, s, v = torch.linalg.svd(posterior_covariance_matrix, full_matrices=False)
        s = s.clip(1e-12, 1e6).real
        posterior_covariance_matrix = u.real @ torch.diag_embed(s) @ v.real
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return torch.distributions.MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix, validate_args=False)


def build_extended_svd(A: torch.tensor):
    U, d, V = torch.linalg.svd(A, full_matrices=True)
    coordinate_mask = torch.ones_like(V[0])
    coordinate_mask[len(d):] = 0
    return U, d, coordinate_mask, V


def generate_measurement_equations(dim, dim_y, mixt):
    A = torch.randn((dim_y, dim))

    u, diag, coordinate_mask, v = build_extended_svd(A)
    diag = torch.sort(torch.rand_like(diag), descending=True).values

    A = u @ (torch.diag(diag) @ v[coordinate_mask == 1, :])
    init_sample = mixt.sample()
    std = (torch.rand((1,)))[0]* max(diag)
    var_observations = std**2

    init_obs = A @ init_sample
    init_obs += torch.randn_like(init_obs) * std
    return A, var_observations, init_obs