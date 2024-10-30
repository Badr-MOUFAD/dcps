import torch
import PIL
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from torch.func import grad
from typing import Tuple
import yaml
from torch.distributions import Distribution
from ddrm.functions.svd_replacement import H_functions
from utils.DiffJPEG.utils import quality_to_factor, diff_round
from utils.DiffJPEG.compress_jpeg import compress_jpeg
from utils.DiffJPEG.decompress_jpeg import decompress_jpeg


class JPEG(H_functions):

    def __init__(self, jpeg_op):
        super(JPEG).__init__()
        self.jpeg = jpeg_op

    def H(self, x):
        """
        x is in [-1, 1]
        """
        return (2 * self.jpeg((x + 1.0) / 2.0) - 1.0).reshape(x.shape[0], -1)

    def H_pinv(self, x):
        return x


class Identity(H_functions):

    def __init__(self):
        super(Identity).__init__()

    def H(self, x):
        return x.reshape(x.shape[0], -1)

    def H_pinv(self, x):
        return x


def compute_cluster_weight(samples, mixt, n_steps=10):
    means = mixt.component_distribution.loc
    cov = mixt.component_distribution.covariance_matrix
    weights = mixt.mixture_distribution.probs

    n_components = means.shape[0]
    n_samples = samples.shape[0]

    def one_step(curr_weights):
        mvn = MultivariateNormal(means, cov)
        joint_logprob = mvn.log_prob(samples.reshape(n_samples, 1, -1))
        joint_logprob += curr_weights.log()
        return joint_logprob.softmax(-1).sum(0)

    curr_weights = weights
    for i in range(n_steps):
        curr_weights = one_step(curr_weights)

    return curr_weights / n_samples


def fwd_mixture(
    means: torch.tensor,
    weights: torch.tensor,
    alphas_cumprod: torch.tensor,
    t: torch.tensor,
    covs: torch.tensor = None,
):
    n_mixtures = weights.shape[0]
    acp_t = alphas_cumprod[t]
    means = acp_t.sqrt() * means
    Id = torch.eye(means.shape[-1])[None, ...].repeat(n_mixtures, 1, 1)
    if covs is None:
        covs = Id
    else:
        covs = (1 - acp_t) * Id + acp_t * covs

    mvn = MultivariateNormal(means, covs)
    return MixtureSameFamily(Categorical(weights), mvn)


class EpsilonNetGM(torch.nn.Module):

    def __init__(self, means, covs, weights, alphas_cumprod):
        super().__init__()
        self.means = means
        self.covs = covs
        self.weights = weights
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x, t):
        grad_logprob = grad(
            lambda x: fwd_mixture(
                self.means, self.covs, self.weights, self.alphas_cumprod, t
            )
            .log_prob(x)
            .sum()
        )
        return -((1 - self.alphas_cumprod[t]) ** 0.5) * grad_logprob(x)


def get_posterior(obs, prior, A, noise_std):
    modified_means = []
    modified_covars = []
    weights = []

    for loc, cov, log_weight in zip(
        prior.component_distribution.loc,
        prior.component_distribution.covariance_matrix,
        prior.mixture_distribution.logits,
    ):
        new_dist = gaussian_posterior(
            obs,
            A,
            torch.zeros_like(obs),
            torch.eye(obs.shape[0]) / (noise_std**2),
            loc,
            cov,
        )
        modified_means.append(new_dist.loc)
        modified_covars.append(new_dist.covariance_matrix)
        prior_x = MultivariateNormal(loc=loc, covariance_matrix=cov)
        log_constant = (
            -torch.linalg.norm(obs - A @ new_dist.loc) ** 2 / (2 * noise_std**2)
            + prior_x.log_prob(new_dist.loc)
            - new_dist.log_prob(new_dist.loc)
        )
        weights.append(log_weight + log_constant)
    weights = torch.tensor(weights)
    # weights = weights / weights.sum()
    weights = weights.softmax(0)
    cat = Categorical(weights)
    ou_norm = MultivariateNormal(
        loc=torch.stack(modified_means, dim=0),
        covariance_matrix=torch.stack(modified_covars, dim=0),
    )
    return MixtureSameFamily(cat, ou_norm)


def gaussian_posterior(
    y, likelihood_A, likelihood_bias, likelihood_precision, prior_loc, prior_covar
):
    prior_precision_matrix = torch.linalg.inv(prior_covar)
    posterior_precision_matrix = (
        prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    )
    posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
    posterior_mean = posterior_covariance_matrix @ (
        likelihood_A.T @ likelihood_precision @ (y - likelihood_bias)
        + prior_precision_matrix @ prior_loc
    )
    # posterior_covariance_matrix += 1e-3 * torch.eye(posterior_covariance_matrix.shape[0])
    posterior_covariance_matrix = (
        posterior_covariance_matrix.T + posterior_covariance_matrix
    ) / 2
    return MultivariateNormal(
        loc=posterior_mean, covariance_matrix=posterior_covariance_matrix
    )


def sliced_wasserstein(dist_1, dist_2, n_slices=100):
    projections = torch.randn(size=(n_slices, dist_1.shape[1])).to(dist_1.device)
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T
    return np.mean(
        [
            wasserstein_distance(u_values=d1.cpu().numpy(), v_values=d2.cpu().numpy())
            for d1, d2 in zip(dist_1_projected, dist_2_projected)
        ]
    )


def posterior_statistics(obs, A, std_y, prior_mean, prior_cov_diag):
    """prior_cov is diagonal with shape batch x dim"""
    dim = prior_mean.shape[-1]
    prior_precision = (1 / prior_cov_diag).reshape(-1, 1, dim) * torch.eye(dim)
    cov = prior_precision + (A.T @ A / std_y**2.0)
    cov_diag = cov.diagonal(dim1=1, dim2=2)
    cov = (1 / cov_diag).reshape(-1, 1, dim) * torch.eye(dim)
    mean = (
        cov
        @ (
            prior_precision @ prior_mean.unsqueeze(-1)
            + (A.T @ obs / std_y**2).unsqueeze(-1)
        )
    ).squeeze(-1)
    return mean, cov


def generate_inpainting(
    anchor_left_top: torch.Tensor,
    sizes: torch.Tensor,
    original_shape: Tuple[int, int, int],
):
    """

    :param anchor_left_top:
    :param sizes:
    :param original_shape: (x, y, n_channels)
    :return:
    """
    A_per_channel = torch.eye(original_shape[0] * original_shape[1])
    mask = torch.ones(original_shape[:2])
    mask[anchor_left_top[0] : anchor_left_top[0] + sizes[0], :][
        :, anchor_left_top[1] : anchor_left_top[1] + sizes[1]
    ] = 0
    return (
        A_per_channel[mask.flatten() == 1, :],
        A_per_channel[mask.flatten() == 0],
        mask,
    )


def display(x, save_path=None, title=None):
    sample = x.squeeze(0).cpu().permute(1, 2, 0)
    sample = (sample + 1.0) * 127.5
    sample = sample.numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path + ".png")


def display_image(x: torch.Tensor, ax=None):
    """Display an image."""
    sample = x.squeeze(0).detach().cpu().permute(1, 2, 0)

    sample = sample.clamp(-1, 1)
    sample = (sample + 1.0) * 127.5
    sample = sample.numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    if ax is None:
        plt.imshow(img_pil)
    else:
        ax.imshow(img_pil)


def display_samples(diff_samples, fwd_mixt, y_obs, A, std_y, t, V=None):
    mixt = fwd_mixt(t)
    mixt_posterior = get_posterior(y_obs, mixt, A, std_y)
    posterior_samples = mixt_posterior.sample((300,))
    if V is not None:
        posterior_samples = (V @ posterior_samples.T).T
        diff_samples = (V @ diff_samples.T).T

    posterior_samples = posterior_samples.cpu()
    diff_samples = diff_samples.detach().cpu()
    alpha = 0.2
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
    ax[0].scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        s=2.0,
        label="posterior samples",
        alpha=alpha,
    )
    ax[0].scatter(
        diff_samples[:, 0], diff_samples[:, 1], s=2.0, label="super_dps", alpha=alpha
    )
    ax[1].scatter(
        posterior_samples[:, 2],
        posterior_samples[:, 3],
        s=2.0,
        label="posterior samples",
        alpha=alpha,
    )
    ax[1].scatter(
        diff_samples[:, 2], diff_samples[:, 3], s=2.0, label="super_dps", alpha=alpha
    )

    for i in [0, 1]:
        ax[i].set_xlim(-25, 25)
        ax[i].set_ylim(-25, 25)
    plt.legend()
    plt.show()


def generate_inverse_problem(
    prior: Distribution, dim: int, std_y: float, A: torch.Tensor = None
):

    if A is None:
        A = torch.randn((1, dim))
    obs = A @ prior.sample() + std_y * torch.randn((A.shape[0],))
    posterior = get_posterior(obs, prior, A, std_y)

    return obs, A, posterior


def check_image(tensor):
    assert (
        torch.max(tensor) <= 1.0 and torch.min(tensor) >= -1.0
    ), "Output images should be (-1, 1.)"


def normalize_tensor(tensor):
    check_image(tensor)
    return (tensor + 1.0) / 2.0


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
