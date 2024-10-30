# %%
import torch, yaml
import numpy as np
from dataclasses import dataclass

from trajnet.models import DDPM
from trajnet.datasets.pedestrians import load_data

from posterior_samplers.cond_sampling_algos import dps, ddrm, pgdm_svd, mcgdiff
from red_diff.models.classifier_guidance_model import ClassifierGuidanceModel, Diffusion
from red_diff.algos.reddiff import REDDIFF
from omegaconf import DictConfig
from posterior_samplers.dcps import dcps
from posterior_samplers.traj_utils import (
    TrajectoryInpainting,
    EpsilonNetSVDTrajectory,
    ProxyEpsilonNet,
)

from local_paths import REPO_PATH

import matplotlib.pyplot as plt


@dataclass
class Config:
    algo = "mcgdiff"
    n_steps = 300
    std = 0.01
    idx_selected = [79, 28]
    n_samples = 5
    device = "cuda:0"


torch.manual_seed(555)
torch.set_default_device(Config.device)

# %%
# load model
model_name = "ucy_20"
epsilon_net = DDPM.load_trajectory_model(
    name=model_name, n_steps=Config.n_steps, device=Config.device
)

len_traj, dim_traj = epsilon_net.net.len_traj, epsilon_net.net.dim_traj

# %%
# load data
_, test_1 = load_data("ucy_student_1")

data_test = torch.vstack((test_1,))

data_test = data_test - data_test.mean(dim=0)
data_test = data_test / data_test.std(dim=0)

# keep xy and reshape
real_trajs = data_test[:, 2:].reshape(-1, len_traj, dim_traj)

# %%
# plotting utils

color = {"generated": "#1f77b4", "real": "#ff7f0e", "observation": "#d62728"}


def plot_traj(ax, traj, color, label=None, pointer_marker=True):
    traj_cpu = traj.cpu()
    traj_x, traj_y = traj_cpu[:, 0], traj_cpu[:, 1]
    marker = "D" if pointer_marker else "."

    ax.plot(
        traj_x,
        traj_y,
        marker=".",
        markersize=4,
        color=color,
        alpha=0.7,
    )
    ax.scatter(
        traj_x[-1],
        traj_y[-1],
        marker=marker,
        color=color,
        alpha=0.7,
    )

    # small hack to get one legend item
    if label:
        ax.scatter(
            traj_x[-1], traj_y[-1], marker=marker, color=color, label=label, alpha=0.7
        )


# %%
# plot traj
fig, ax = plt.subplots()
idx_selected = Config.idx_selected

for idx in idx_selected:
    # select traj
    selected_traj = real_trajs[idx]
    plot_traj(ax, selected_traj, color=color["real"])

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal", adjustable="box")

# %%
# create masks

# mask last 5 steps
mask_1 = TrajectoryInpainting(
    len_traj,
    dim_traj,
    missing_coordinates=torch.tensor([15, 16, 17, 18, 19]),
    device=Config.device,
)

# observe last 5 steps
mask_2 = TrajectoryInpainting(
    len_traj, dim_traj, missing_coordinates=torch.arange(15), device=Config.device
)

# observe [8, 13] steps
mask_3 = TrajectoryInpainting(
    len_traj,
    dim_traj,
    missing_coordinates=torch.tensor(list(range(8)) + list(range(13, 20))),
    device=Config.device,
)

all_masks = [mask_1, mask_2, mask_3]


# %%
# inverse prob
selected_trajs = real_trajs[idx_selected]

all_reconstructions = []
for mask in all_masks:
    reconstructions = torch.zeros(
        (len(selected_trajs), Config.n_samples, len_traj, dim_traj)
    )

    for i, traj in enumerate(selected_trajs):
        sigma_y = Config.std
        obs = mask.H(traj[None])
        obs = obs + sigma_y * torch.randn_like(obs)

        initial_noise = torch.randn(Config.n_samples, len_traj, dim_traj)

        if Config.algo == "dcps":
            L = 4
            n_steps = Config.n_steps // (L - 1)
            obs_timesteps = torch.linspace(0, len(epsilon_net.alphas_cumprod) - 1, L)

            reconstructions[i] = dcps(
                initial_noise=initial_noise,
                epsilon_net=epsilon_net,
                obs=obs,
                A=mask.H,
                obs_std=sigma_y,
                n_steps=n_steps,
                obs_timesteps=obs_timesteps,
                optimizer="SGD",
                learning_rate=5e-2,
                langevin_steps=5,
            )
        elif Config.algo == "ddrm":
            inverse_problem = (obs, mask, sigma_y)
            samples = ddrm(
                initial_noise.unsqueeze(1),
                ProxyEpsilonNet(epsilon_net.net),
                inverse_problem,
                epsilon_net.timesteps,
                epsilon_net.alphas_cumprod,
                Config.device,
            )
            reconstructions[i] = samples.squeeze(1)
        elif Config.algo == "dps":
            inverse_problem = (obs, mask.H, sigma_y)
            reconstructions[i] = dps(
                initial_noise, inverse_problem, epsilon_net, gamma=5e-2
            )
        elif Config.algo == "reddiff":
            with open(
                REPO_PATH / "./src/red_diff/_configs/algo/reddiff.yaml", "r"
            ) as conf:
                reddiff_cfg = DictConfig(yaml.safe_load(conf))

            clfg = ClassifierGuidanceModel(
                model=ProxyEpsilonNet(epsilon_net.net),
                classifier=None,
                diffusion=Diffusion(device=Config.device),
                cfg=None,
            )
            reddiff_cfg = DictConfig({"algo": reddiff_cfg})
            rdiff = REDDIFF(clfg, cfg=reddiff_cfg, H=mask)
            samples = rdiff.sample(
                initial_noise.unsqueeze(1),
                None,
                epsilon_net.timesteps,
                y_0=obs.reshape(1, -1),
            ).detach()
            reconstructions[i] = samples.squeeze(1)
        elif Config.algo == "pgdm":
            epsilon_net_svd = EpsilonNetSVDTrajectory(
                net=epsilon_net.net,
                alphas_cumprod=epsilon_net.alphas_cumprod,
                timesteps=epsilon_net.timesteps,
                H_func=mask,
                len_traj=len_traj,
                dim_traj=dim_traj,
                device=Config.device,
            )
            reconstructions[i] = pgdm_svd(
                initial_noise, epsilon_net_svd, obs, mask, sigma_y
            )
        elif Config.algo == "mcgdiff":
            coordinates_mask = torch.isin(
                torch.arange(len_traj * dim_traj, device=Config.device),
                torch.arange(mask.kept_indices.shape[0], device=Config.device),
            )

            epsilon_net_svd = EpsilonNetSVDTrajectory(
                net=epsilon_net.net,
                alphas_cumprod=epsilon_net.alphas_cumprod,
                timesteps=epsilon_net.timesteps,
                H_func=mask,
                len_traj=len_traj,
                dim_traj=dim_traj,
                device=Config.device,
            )

            reconstructions[i] = mcgdiff(
                initial_noise=torch.randn((1000, len_traj, dim_traj)),
                epsilon_net=epsilon_net_svd,
                obs=obs,
                H_func=mask,
                coordinates_mask=coordinates_mask,
                std_obs=sigma_y,
                device=Config.device,
                n_return=Config.n_samples,
            )

    all_reconstructions.append(reconstructions.cpu())

# %%
# plotting

fig, axes = plt.subplots(1, len(all_masks), sharey=True)

for i, mask in enumerate(all_masks):
    ax = axes[i]
    reconstructions_cpu = all_reconstructions[i]

    # plot reconstructions
    label = "generated"
    for j, traj in enumerate(reconstructions_cpu.reshape(-1, len_traj, dim_traj)):
        add_label = j == 0 and i == 1
        plot_traj(
            ax,
            traj,
            color=color[label],
            label=label if add_label else None,
        )

    for j, traj in enumerate(selected_trajs.reshape(-1, len_traj, dim_traj)):
        # plot real
        label = "real"
        add_label = j == 0 and i == 1

        plot_traj(
            ax,
            traj,
            color=color[label],
            label=label if add_label else None,
        )

        # plot observation
        label = "observation"
        # NOTE this works only for contiguous masking
        masked_traj = mask.H(traj[None])
        masked_traj = masked_traj.reshape(dim_traj, -1).permute(1, 0)

        plot_traj(
            ax,
            masked_traj,
            color=color[label],
            label=label if add_label else None,
            pointer_marker=False,
        )

#################
# just skip: make a commun legend ###
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# keep unique
_, indices = np.unique(labels, return_index=True)
lines = [lines[i] for i in indices]
labels = [labels[i] for i in indices]
# sort to put skglm first
indices = np.argsort(labels)[::-1]
lines = [lines[i] for i in indices]
labels = [labels[i] for i in indices]

# create legend
legend = fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.1),  # position on the top
    # bbox_to_anchor=(0.52, 0.08),  # position in the bottom
    ncol=3,
    columnspacing=1.5,
)
################

# fig.set_size_inches(3.2, 7.8)
fig.tight_layout()
axes[1].set_title(Config.algo)


# %%
