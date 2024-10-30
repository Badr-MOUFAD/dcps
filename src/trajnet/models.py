from typing import List

import torch
from torch import nn
import matplotlib.pyplot as plt

from trajnet.denoisers import EpsilonNet
from trajnet.diffusion_utils import EpsNet
from trajnet.schedulers import get_schedulers

from local_paths import LARGE_FILE_DIR


PATH_CHECKPOINTS = LARGE_FILE_DIR / "trajectories/checkpoints/"


class DDPM(nn.Module):

    def __init__(
        self,
        denoiser: EpsilonNet,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device

        self.denoiser = denoiser
        self.dim_traj, self.len_traj = denoiser.dim_traj, denoiser.len_traj

        self.n_diffusion_steps = len(denoiser.alpha_cum_prod)
        self.mean_schedule, self.std_schedule, self.diffusion_rate = get_schedulers(
            denoiser.alpha_cum_prod
        )

        # put into device
        self.to(device)

    def _forward_diffusion(
        self, batch_trajectories: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Perform the forward diffusion process.

        Return
        ------
        noisy_traj : Tensor, shape (batch_size, len_traj, dim_traj)
            Noisy version of trajectories.
        """
        # `batch_trajectories`, `noise` have shape (batch_size, len_traj, dim_traj)
        # `t` is a 1 dim tensor of shape (1,) or (batch_size,)
        mean_schedule = self.mean_schedule[t][:, None, None]
        std_schedule = self.std_schedule[t][:, None, None]

        return mean_schedule * batch_trajectories + std_schedule * noise

    def compute_batch_loss(self, batch_trajectories):
        device = self.device
        batch_trajectories = batch_trajectories.to(device)

        time_steps = torch.randint(
            0, self.n_diffusion_steps, size=(len(batch_trajectories),), device=device
        )
        noise = torch.randn_like(batch_trajectories, device=device)

        noisy_trajs = self._forward_diffusion(batch_trajectories, noise, time_steps)
        predicted_noise = self.denoiser.predict_noise(noisy_trajs, time_steps)

        loss_per_sample = ((noise - predicted_noise) ** 2).sum(dim=(-2, -1)).mean()

        return loss_per_sample

    @torch.no_grad()
    def sample(self, n_samples=100):
        self.eval()
        device = self.device
        trajectories = torch.randn(
            (n_samples, self.len_traj, self.dim_traj), device=self.device
        )

        for t in reversed(torch.arange(1, self.n_diffusion_steps, device=device)):
            noise = torch.randn_like(trajectories, device=self.device)

            # paper Ho et al. 2020 uses sigma_t = beta_t
            scale_noise = torch.sqrt(self.diffusion_rate[t])
            trajectories = self.denoiser.denoise(trajectories, t) + scale_noise * noise

        # last denoising
        trajectories = self.denoiser.denoise(trajectories, 0)

        return trajectories

    @classmethod
    def load_from_pretrained(cls, path: str, device: str):
        model = torch.load(path, map_location=device)
        # FIXME homogenize how models are saved
        denoiser = getattr(model, "denoiser", model)

        return cls(denoiser=denoiser, device=device)

    @classmethod
    def load_trajectory_model(
        cls, name: str = "dummy_20", n_steps: int = 300, device: str = "cpu"
    ) -> EpsNet:
        """Load trajectory model.

        It returns an EpsNet that is compatible with dcps.

        Parameters
        ----------
        name : std, default="dummy_20"
            The name of the model. It must be either of these:
                - "ucy_20"

        n_steps : int, default=300
            The number of diffusion steps to use

        device : str, default="cpu"
            The device where to put the model.

        Return
        ------
        EpsNet : EpsNet
            Epsilon network compatible with dcps algorithm
        """
        DICT_MODELS = {
            "ucy_20": PATH_CHECKPOINTS / "ucy_len_20_n_diff_steps_1000.pt",
        }

        # load model and put it in `eval` mode
        selected_model = DICT_MODELS.get(name, None)
        model = cls.load_from_pretrained(selected_model, device)
        model.eval()
        model.requires_grad_(False)

        # build Epsnet
        alphas_cumprod = model.denoiser.alpha_cum_prod

        timesteps = torch.linspace(
            0, len(alphas_cumprod) - 1, n_steps, device=device
        ).long()
        epsilon_net = EpsNet(
            net=model.denoiser, alphas_cumprod=alphas_cumprod, timesteps=timesteps
        )

        return epsilon_net

    # for debugging purposes
    @torch.no_grad()
    def _plot_traj(self, trajectories: torch.Tensor, ax=None, n_skip=20):
        # trajectories has shape (batch_size, len_traj, dim_traj)
        # ax is plt object where to plot

        ax = ax if ax is not None else plt
        color = "#1f77b4"

        for i in range(0, len(trajectories), n_skip):
            traj = trajectories[i]
            traj_x, traj_y = traj[:, -2], traj[:, -1]

            ax.scatter(traj_x, traj_y, color=color, marker=".", s=2)
            ax.scatter(traj_x[-1], traj_y[-1], marker="D", color=color, s=8)

    @torch.no_grad()
    def _plot_forward_diffusion(
        self,
        batch_trajectories: torch.Tensor,
        diffusion_steps: List[int],
        n_skip: int = 20,
    ):
        batch_trajectories = batch_trajectories.to(self.device)
        n_samples = len(batch_trajectories)

        fig, axes = plt.subplots(1, len(diffusion_steps), sharey=True)

        for i, t in enumerate(diffusion_steps):

            noise = torch.randn(
                (n_samples, self.len_traj, self.dim_traj),
                device=self.device,
            )

            time_step = torch.tensor([t], dtype=int, device=self.device)
            noisy_traj = self._forward_diffusion(
                batch_trajectories,
                noise,
                time_step,
            )
            # transfer to cpu
            noisy_traj = noisy_traj.cpu()

            # plot
            self._plot_traj(noisy_traj, axes[i], n_skip)
            axes[i].set_title(f"t={t}")

            axes[i].set_aspect("equal", adjustable="box")
            # uncomment to remove x and y ticks
            # axes[i].set_xticks([], [])
            # axes[i].set_yticks([], [])

        plt.tight_layout()
        plt.show()
