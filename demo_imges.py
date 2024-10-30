# %%
import time
from dataclasses import dataclass

from posterior_samplers.cond_sampling_algos import mcgdiff
from red_diff.algos.reddiff import REDDIFF
from omegaconf import DictConfig
from utils.DiffJPEG.DiffJPEG import DiffJPEG

import yaml
from posterior_samplers.dcps import dcps
from posterior_samplers.cond_sampling_algos import dps, ddrm, pgdm_svd, pgdm_jpeg
from utils.utils import display, JPEG, Identity
from posterior_samplers.diffusion_utils import (
    load_epsilon_net,
    EpsilonNetSVD,
)
from utils.metrics import LPIPS, PSNR, SSIM
from PIL import Image
from red_diff.models.classifier_guidance_model import ClassifierGuidanceModel, Diffusion
import numpy as np
import torch

import math
import matplotlib.pyplot as plt

from local_paths import REPO_PATH, LARGE_FILE_DIR


IMAGING_DIR = LARGE_FILE_DIR
torch.manual_seed(48)


@dataclass
class Config:
    model = "ffhq"
    n_steps = 300
    algo = "dcps"
    im_idx = "00010.png"
    task = "inpainting_middle"
    noise_type = "poisson"
    std = 0.01
    poisson_rate = 0.05
    n_samples = 1
    device = "cuda:1"


args = Config
device = args.device
torch.set_default_device(device)
torch.cuda.empty_cache()
print(f"std: {args.std}")


task_name = args.task

n_steps = args.n_steps
epsilon_net = load_epsilon_net(args.model, n_steps=n_steps, device=device)


image = Image.open(IMAGING_DIR / f"{args.model}/validation_set/{args.im_idx}")

im = torch.tensor(np.array(image)).type(torch.FloatTensor).to(device)
x_origin = ((im - 127.5) / 127.5).squeeze(0)

D_OR = x_origin.shape
if len(D_OR) == 2:
    D_OR = (1,) + D_OR
    x_origin = x_origin.reshape(*D_OR)
else:
    D_OR = D_OR[::-1]
    x_origin = x_origin.permute(2, 0, 1)

D_FLAT = math.prod(D_OR)
eta = 1
sigma_y = args.std

if task_name.startswith("jpeg"):
    ip_type = "jpeg"
    jpeg_quality = int(task_name.replace("jpeg", ""))

    operator = DiffJPEG(
        height=256, width=256, differentiable=True, quality=jpeg_quality
    ).to(device)
    H_funcs = JPEG(operator)
    y_0 = H_funcs.H(x_origin.unsqueeze(0)).reshape(1, 3, 256, 256)

elif task_name == "denoising":
    ip_type = "linear"
    H_funcs = Identity()
    y_0 = H_funcs.H(x_origin.unsqueeze(0)).reshape(1, 3, 256, 256)

else:
    ip_type = "linear"
    H_funcs = torch.load(
        IMAGING_DIR / f"masks_img256/{task_name}.pt",
        map_location=device,
    )
    operator = H_funcs.H

    epsilon_net_svd = EpsilonNetSVD(
        net=epsilon_net.net,
        alphas_cumprod=epsilon_net.alphas_cumprod,
        timesteps=epsilon_net.timesteps,
        H_func=H_funcs,
        device=device,
    )

    if task_name == "sr4":
        ratio = 4
        D_OBS = (D_OR[0], int(D_OR[1] / ratio), int(D_OR[2] / ratio))
        y_0 = H_funcs.H(x_origin[None, ...]).reshape(*D_OBS)

    elif task_name == "sr16":
        ratio = 16
        D_OBS = (D_OR[0], int(D_OR[1] / ratio), int(D_OR[2] / ratio))
        y_0 = H_funcs.H(x_origin[None, ...]).reshape(*D_OBS)
    elif task_name in ["outpainting_half", "inpainting_middle", "outpainting_expand"]:
        y_0 = H_funcs.H(x_origin[None, ...])

        y_0_img = -torch.ones(math.prod(D_OR), device=y_0.device)
        y_0_img[: y_0.shape[-1]] = y_0[0]
        y_0_img = H_funcs.V(y_0_img[None, ...])
        y_0_img = y_0_img.reshape(*D_OR)

if args.noise_type == "gaussian":
    y_0 = (y_0 + sigma_y * torch.randn_like(y_0)).clip(-1.0, 1.0)

elif args.noise_type == "poisson":
    rate = args.poisson_rate
    y_0 = torch.poisson(rate * ((y_0 + 1.0) / 2.0) * 255.0).clip(0, rate * 255.0)
    y_0 = 2 * (y_0 / (rate * 255.0)) - 1.0

# plot
if task_name in ["outpainting_half", "inpainting_middle", "outpainting_expand"]:
    y_0_img = -torch.ones(math.prod(D_OR), device=y_0.device)
    y_0_img[: y_0.shape[-1]] = y_0[0]
    y_0_img = H_funcs.V(y_0_img[None, ...])
    y_0_img = y_0_img.reshape(*D_OR)
else:
    y_0_img = y_0


display(y_0_img.detach().cpu(), title="Observation")
display(x_origin.cpu(), title="Ground-truth")


ddrm_timesteps = epsilon_net.timesteps.clone()
ddrm_timesteps[-1] = ddrm_timesteps[-1] - 1


initial_noise = torch.randn(args.n_samples, *D_OR)

lpips, ssim, psnr = LPIPS(), SSIM(), PSNR()

start = time.time()

if args.algo == "dps":
    if args.noise_type == "gaussian":
        samples = dps(
            initial_noise,
            (y_0, H_funcs.H, sigma_y),
            epsilon_net,
            gamma=1.0,
            noise_type="gaussian",
        ).clamp(-1, 1)
    elif args.noise_type == "poisson":
        samples = dps(
            initial_noise,
            (y_0, H_funcs.H, y_0),
            epsilon_net,
            gamma=0.3,
            noise_type="poisson",
            poisson_rate=args.poisson_rate,
        ).clamp(-1, 1)

elif args.algo == "dcps":
    L = 4
    n_steps = n_steps // (L - 1)
    obs_timesteps = torch.linspace(0, 999, L)
    samples = dcps(
        initial_noise=initial_noise,
        epsilon_net=epsilon_net,
        ip_type=ip_type,
        obs=y_0.reshape(1, -1),
        A=H_funcs.H,
        obs_std=sigma_y,
        n_steps=n_steps,
        obs_timesteps=obs_timesteps,
        optimizer="SGD",
        gradient_steps=2,
        learning_rate=1.5,
        langevin_steps=5,
        gamma=1e-3,
        poisson_rate=args.poisson_rate,
        noise_type=args.noise_type,
    ).clamp(-1, 1)

elif args.algo == "mcgdiff":
    coordinates_mask = H_funcs.singulars() != 0

    if args.task == "outpainting_half" or args.task == "outpainting_expand":
        coordinates_mask = torch.isin(
            torch.arange(math.prod(D_OR), device=H_funcs.kept_indices.device),
            torch.arange(
                H_funcs.kept_indices.shape[0], device=H_funcs.kept_indices.device
            ),
        )

    elif args.task == "inpainting_middle":
        coordinates_mask = torch.isin(
            torch.arange(math.prod(D_OR), device=H_funcs.kept_indices.device),
            torch.arange(
                H_funcs.kept_indices.shape[0], device=H_funcs.kept_indices.device
            ),
        )

    elif args.task == "sr4" or args.task == "sr16":
        coordinates_mask = torch.cat(
            (
                coordinates_mask,
                torch.tensor([0] * (torch.tensor(D_OR).prod() - len(coordinates_mask))),
            )
        )

    samples = mcgdiff(
        initial_noise,
        epsilon_net_svd,
        y_0.reshape(1, -1),
        H_funcs,
        coordinates_mask,
        sigma_y,
        device,
    ).clamp(-1, 1)

elif args.algo == "ddrm":
    inverse_problem = (y_0, H_funcs, sigma_y)
    samples = ddrm(
        initial_noise,
        epsilon_net.net,
        inverse_problem,
        epsilon_net.timesteps,
        epsilon_net.alphas_cumprod,
        args.device,
    ).clamp(-1, 1)

elif args.algo == "pgdm":
    if args.task == "jpeg":
        samples = pgdm_jpeg(
            initial_noise, epsilon_net, y_0.reshape(1, -1), H_funcs, sigma_y
        ).clamp(-1, 1)
    else:
        samples = pgdm_svd(
            initial_noise, epsilon_net_svd, y_0.reshape(1, -1), H_funcs, sigma_y
        ).clamp(-1, 1)

elif args.algo == "reddiff":
    with open(REPO_PATH / "src/red_diff/_configs/algo/reddiff.yaml", "r") as conf:
        reddiff_cfg = DictConfig(yaml.safe_load(conf))

    clfg = ClassifierGuidanceModel(
        model=epsilon_net.net,
        classifier=None,
        diffusion=Diffusion(device=device),
        cfg=None,
    )
    reddiff_cfg = DictConfig({"algo": reddiff_cfg})
    rdiff = REDDIFF(clfg, cfg=reddiff_cfg, H=H_funcs)
    samples = (
        rdiff.sample(initial_noise, None, epsilon_net.timesteps, y_0=y_0.reshape(1, -1))
        .clamp(-1, 1)
        .detach()
    )

print(f"runtime {(time.time() - start):.2f} s")

for i in range(args.n_samples):
    display(samples[i].clamp(-1, 1), title=f"reconstruction {i}")
    plt.show()

print(f"{args.algo} metrics")
print(f"lpips: {lpips.score(samples, x_origin)}")
print(f"ssim: {ssim.score(samples, x_origin)}")
print(f"psnr: {psnr.score(samples, x_origin)}")
