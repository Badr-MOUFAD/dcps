import torch
import numpy as np
from scipy.stats import wasserstein_distance


def compute_len_trajectories(batch_trajs: torch.Tensor) -> torch.Tensor:
    r"""Compute length of a batch of trajectories.

    The length of a trajectory x_1, ..., x_n is defined as

    ..math::

        L = \sum_i || x_i - x_{i-1} ||

    Parameters
    ----------
    batch_trajs: Tensor, shape (batch_size, len_traj, dim_traj)

    Return
    ------
    len_trajs : Tensor, shape (batch_size,)
    """
    diff_trajs = torch.diff(batch_trajs, dim=-2)
    len_trajs_segments = (diff_trajs**2).sum(dim=-1)
    len_trajs = torch.sqrt(len_trajs_segments).sum(dim=-1)

    return len_trajs


def sliced_wasserstein(
    dist_1: torch.Tensor, dist_2: torch.Tensor, n_slices: int = 100
) -> float:
    """Compute sliced Wasserstein distance between two distributions.

    Assumes that both ``dist_1`` and ``dist_2`` have the same dimension.

    Parameters
    ----------
    dist_1 : Tensor

    dist_2 : Tensor

    n_slices : int, default=100
        The number of the considered random projections.

    Return
    ------
    sw_distance : float
    """
    if dist_1.ndim > 2:
        dist_1 = dist_1.reshape(dist_1.shape[0], -1)
        dist_2 = dist_2.reshape(dist_2.shape[0], -1)

    projections = torch.randn(size=(n_slices, dist_1.shape[1]), device=dist_1.device)
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T

    dist_1_projected = dist_1_projected.cpu().numpy()
    dist_2_projected = dist_2_projected.cpu().numpy()

    return np.mean(
        [
            wasserstein_distance(u_values=d1, v_values=d2)
            for d1, d2 in zip(dist_1_projected, dist_2_projected)
        ]
    )
