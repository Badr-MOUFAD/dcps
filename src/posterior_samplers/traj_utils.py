from typing import List

import torch
from ddrm.functions.svd_replacement import H_functions
from posterior_samplers.diffusion_utils import EpsilonNet


class TrajectoryInpainting(H_functions):
    """Mask to define Trajectory inpainting problems.

    The ``missing_coordinates`` are the missing coordinates in the timeserie.
    Example of trajectory of dimension 2: when ``missing_coordinates=[0, 5]``,
    it will be interpreted as ``[(x_0, y_0), (x_5, y_5)]``.

    Note
    ----
    - Negative indexing is not supported in ``missing_coordinates``.
    - All methods operate on batches, shape ``(batch_size, len_traj, dim_traj)``.
    """

    def __init__(
        self,
        len_traj: int,
        dim_traj: int,
        missing_coordinates: List[int],
        device: str = "cpu",
    ):
        self.len_traj, self.dim_traj = len_traj, dim_traj

        self._singulars = torch.ones(
            len_traj * dim_traj - 2 * len(missing_coordinates), device=device
        )
        self.missing_indices = torch.tensor(
            [i + j * len_traj for i in missing_coordinates for j in range(dim_traj)],
            device=device,
        ).long()
        self.kept_indices = torch.tensor(
            [i for i in range(len_traj * dim_traj) if i not in self.missing_indices],
            device=device,
        ).long()

    def V(self, vec: torch.Tensor):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, : self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0] :]
        return (
            out.reshape(vec.shape[0], self.dim_traj, self.len_traj)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )

    # NOTE the `for_H` argument is for compatibility with mcgdiff algorithm
    def Vt(self, vec: torch.Tensor, for_H=True):
        temp = (
            vec.clone()
            .reshape(vec.shape[0], self.len_traj, self.dim_traj)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )
        out = torch.zeros_like(temp)
        out[:, : self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0] :] = temp[:, self.missing_indices]
        return out

    def U(self, vec: torch.Tensor):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec: torch.Tensor):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec: torch.Tensor):
        temp = torch.zeros(
            (vec.shape[0], self.len_traj * self.dim_traj), device=vec.device
        )
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


class ProxyEpsilonNet(torch.nn.Module):
    """Proxy to treat trajectories as images.

    This class is meant to ensure compatibility with ddrm, reddiff algorithms
    for trajectories.
    """

    def __init__(self, epsnet):
        super().__init__()
        self.net = epsnet

    def forward(self, x: torch.Tensor, t: int):
        if len(x.shape) == 4:
            x = x.squeeze(1)

        if not isinstance(t, int):
            t = t.int()

        return self.net.forward(x, t).unsqueeze(1)


class EpsilonNetSVDTrajectory(EpsilonNet):
    """Proxy for compatibility with pdgm algorithm run on trajectories."""

    def __init__(
        self, net, alphas_cumprod, timesteps, H_func, len_traj, dim_traj, device="cuda"
    ):
        super().__init__(net, alphas_cumprod, timesteps)
        self.len_traj, self.dim_traj = len_traj, dim_traj
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device
        self.len_traj = len_traj
        self.dim_traj = dim_traj

    def forward(self, x, t):
        shape = (x.shape[0], self.len_traj, self.dim_traj)
        x = self.H_func.V(x.to(self.device)).reshape(shape)
        return self.H_func.Vt(self.net(x, t))


if __name__ == "__main__":
    len_traj, dim_traj = 20, 2
    missing_coordinates = [0, 10, 19]

    mask = TrajectoryInpainting(len_traj, dim_traj, missing_coordinates, device="cpu")
    traj = torch.arange(len_traj * dim_traj).reshape(1, len_traj, dim_traj)

    out = mask.H(traj)

    mask.H_pinv(out).reshape(1, len_traj, dim_traj)
