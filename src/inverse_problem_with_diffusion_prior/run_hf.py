import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu/'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=70'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import math
import sys

from diffusers import DDPMPipeline
from inverse_problem_with_diffusion_prior.ipwdp.svd_replacement import Deblurring2D, SuperResolution, Inpainting, Colorization
from ddrm.functions.denoising import efficient_generalized_steps
import torch
import numpy as np
from inverse_problem_with_diffusion_prior.ipwdp.image_utils import display_sample
from inverse_problem_with_diffusion_prior.ipwdp.generative_models import ScoreModel
from inverse_problem_with_diffusion_prior.ipwdp.optimal_particle_filter import particle_filter
import matplotlib.pyplot as plt
from dps.guided_diffusion.gaussian_diffusion import create_sampler
from dps.guided_diffusion.condition_methods import PosteriorSampling
from dps.guided_diffusion.measurements import NonLinearOperator, get_noise
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os


def display_black_and_white(img):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.imshow(img)
    return fig


def find_furthest_particles_in_clound(particles, N=None):
    N = particles.shape[0]
    dist_matrix = torch.cdist(particles.reshape(N, -1), particles.reshape(N, -1), p=2)
    return (dist_matrix==torch.max(dist_matrix)).nonzero()[0]


class DPSHfuncOperator(NonLinearOperator):
    def __init__(self, H_func, dim, **kwargs):
        super().__init__()
        self.H_func = H_func
        self.dim = dim

    def forward(self, data, **kwargs):
        return self.H_func.H(data).reshape(data.shape[0], *self.dim)


class EpsilonNetSVD(torch.nn.Module):

    def __init__(self, H_funcs, unet, dim):
        super().__init__()
        self.unet = unet
        self.H_funcs = H_funcs
        self.dim = dim

    def forward(self, x, t):
        x_normal_basis = self.H_funcs.V(x).reshape(-1, *self.dim)
        #x_normal_basis = x.reshape(-1, 1, 28, 28)
        t_emb = torch.tensor(t).to(x.device)#.repeat(x.shape[0]).to(x.device)
        eps = self.unet(x_normal_basis, t_emb).sample
        #eps_svd_basis = eps.reshape(x.shape[0], -1)
        #eps = eps - .5
        eps_svd_basis = self.H_funcs.Vt(eps, for_H=False)
        return eps_svd_basis


class EpsilonNetDDRM(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        return self.unet(x, t).sample


def load_hf_model(config_hf):
    pipeline = DDPMPipeline.from_pretrained(config_hf.hf_model_tag).to('cuda:0')
    all_images = pipeline(batch_size=1)
    image = all_images.images[0]
    x_origin = ((torch.tensor(np.array(image)).type(torch.FloatTensor).cuda() - 127.5) / 127.5)

    D_OR = x_origin.shape
    if len(D_OR) == 2:
        D_OR = (1, ) + D_OR
        x_origin = x_origin.reshape(*D_OR)
    else:
        D_OR = D_OR[::-1]
        x_origin = x_origin.permute(2, 0, 1)
    D_FLAT = math.prod(D_OR)
    return pipeline, x_origin, D_OR, D_FLAT


def plot(x):
    if x.shape[0] == 1:
        fig = display_black_and_white(x[0].cpu())
    else:
        fig = display_sample(x.cpu())
    return fig


def load_operator(task_cfg, D_OR, x_origin):
    sigma_y = task_cfg.sigma_y
    if task_cfg.name == 'deblur_2d':
        kernel_size = math.ceil(D_OR[2] * task_cfg.kernel_size) * (3 // D_OR[0])
        sigma = math.ceil(D_OR[2] * task_cfg.kernel_std)
        pdf = lambda x: torch.exp(-0.5 * (x / sigma) ** 2)
        kernel1 = pdf(torch.arange(-kernel_size, kernel_size + 1)).cuda()
        kernel2 = pdf(torch.arange(-kernel_size, kernel_size + 1)).cuda()
        kernel1 = kernel1 / kernel1.sum()
        kernel2 = kernel2 / kernel2.sum()

        H_funcs = Deblurring2D(kernel1,
                               kernel2,
                               D_OR[0],
                               D_OR[1], 0)


        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0_origin = y_0_origin.reshape(*D_OR)
        y_0 = y_0_origin + sigma_y * torch.randn_like(y_0_origin)
        y_0_img = y_0
        diag = H_funcs.singulars()
        coordinates_mask = diag != 0
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten()[coordinates_mask].cpu()
        diag = diag[coordinates_mask].cpu()
        D_OBS = D_OR

    elif task_cfg.name == 'super_resolution':
        ratio = task_cfg.ratio
        H_funcs = SuperResolution(channels=D_OR[0], img_dim=D_OR[2], ratio=ratio, device='cuda:0')
        D_OBS = (D_OR[0], int(D_OR[1] / ratio), int(D_OR[2] / ratio))
        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0_origin = y_0_origin.reshape(*D_OBS)
        y_0 = (y_0_origin + sigma_y * torch.randn_like(y_0_origin)).clip(-1., 1.)
        y_0_img = y_0

        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten().cpu()
        diag = H_funcs.singulars()
        coordinates_mask = diag != 0
        coordinates_mask = torch.cat(
            (coordinates_mask, torch.tensor([0] * (torch.tensor(D_OR).prod() - len(coordinates_mask))).cuda()))

    elif task_cfg.name == 'outpainting':
        center, width, height = task_cfg.center, task_cfg.width, task_cfg.height
        range_width = (math.floor((center[0] - width / 2)*D_OR[1]), math.ceil((center[0] + width / 2)*D_OR[1]))
        range_height = (math.floor((center[1] - height / 2)*D_OR[2]), math.ceil((center[1] + width / 2)*D_OR[2]))
        mask = torch.ones(*D_OR[1:])
        mask[range_width[0]: range_width[1], range_height[0]:range_height[1]] = 0
        missing_r = torch.nonzero(mask.flatten()).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)

        H_funcs = Inpainting(channels=D_OR[0], img_dim=D_OR[1], missing_indices=missing, device=x_origin.device)
        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0 = (y_0_origin + sigma_y * torch.randn_like(y_0_origin)).clip(-1., 1.)
        y_0_img = -torch.ones(math.prod(D_OR), device=y_0.device)
        y_0_img[:y_0.shape[-1]] = y_0[0]
        y_0_img = H_funcs.V(y_0_img[None, ...])
        y_0_img = y_0_img.reshape(*D_OR)
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten().cpu()
        diag = H_funcs.singulars()
        coordinates_mask = torch.isin(torch.arange(math.prod(D_OR),
                                                   device=H_funcs.kept_indices.device),
                                      torch.arange(H_funcs.kept_indices.shape[0],
                                                   device=H_funcs.kept_indices.device))
        D_OBS = (math.prod(D_OR) - len(missing),)
    elif task_cfg.name == 'inpainting':
        center, width, height = task_cfg.center, task_cfg.width, task_cfg.height
        range_width = (math.floor((center[0] - width / 2)*D_OR[1]), math.ceil((center[0] + width / 2)*D_OR[1]))
        range_height = (math.floor((center[1] - height / 2)*D_OR[2]), math.ceil((center[1] + width / 2)*D_OR[2]))
        mask = torch.zeros(*D_OR[1:])
        mask[range_width[0]: range_width[1], range_height[0]:range_height[1]] = 1
        missing_r = torch.nonzero(mask.flatten()).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)

        H_funcs = Inpainting(channels=D_OR[0], img_dim=D_OR[1], missing_indices=missing, device=x_origin.device)
        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0 = (y_0_origin + sigma_y * torch.randn_like(y_0_origin)).clip(-1., 1.)
        y_0_img = -torch.ones(math.prod(D_OR), device=y_0.device)
        y_0_img[:y_0.shape[-1]] = y_0[0]
        y_0_img = H_funcs.V(y_0_img[None, ...])
        y_0_img = y_0_img.reshape(*D_OR)
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten().cpu()
        diag = H_funcs.singulars()
        coordinates_mask = torch.isin(torch.arange(math.prod(D_OR),
                                                   device=H_funcs.kept_indices.device),
                                      torch.arange(H_funcs.kept_indices.shape[0],
                                                   device=H_funcs.kept_indices.device))
        D_OBS = (math.prod(D_OR) - len(missing),)
    elif task_cfg.name == 'colorization':

        H_funcs = Colorization(D_OR[1], x_origin.device)

        y_0_origin = H_funcs.H(x_origin[None, ...])
        y_0 = y_0_origin + sigma_y * torch.randn_like(y_0_origin)
        y_0_img = H_funcs.H_pinv(y_0_origin).reshape(D_OR)
        diag = H_funcs.singulars()
        coordinates_mask = diag != 0
        U_t_y_0 = H_funcs.Ut(y_0[None, ...]).flatten()[coordinates_mask].cpu()
        diag = diag[coordinates_mask].cpu()
        coordinates_mask = torch.cat(
            (coordinates_mask, torch.tensor([0] * (torch.tensor(D_OR).prod() - len(coordinates_mask))).cuda()))
        D_OBS = (y_0.shape[-1],)
    else:
        raise NotImplementedError

    return H_funcs, y_0, y_0_origin, y_0_img, U_t_y_0, diag, coordinates_mask, D_OBS


def run_mcg_diff(mcg_diff_config, score_model, n_max_gpu, dim, U_t_y_0, diag, coordinates_mask, sigma_y, timesteps, eta, H_funcs):
    total_N = mcg_diff_config.N_total
    n_particles = mcg_diff_config.N_particles

    particles_mcg_diff = []
    for j in enumerate(range(total_N)):
        batch_initial_particles = torch.randn(size=(n_particles, dim))
        particles = particle_filter(initial_particles=batch_initial_particles.cpu(),
                                    observation=U_t_y_0,
                                    likelihood_diagonal=diag.cpu(),
                                    score_model=score_model,
                                    coordinates_mask=coordinates_mask.cpu(),
                                    var_observation=sigma_y**2,
                                    timesteps=timesteps.cpu(),
                                    eta=eta,
                                    n_samples_per_gpu_inference=n_max_gpu,
                                    gaussian_var=mcg_diff_config.gaussian_var)
        H_funcs = H_funcs.to("cpu")
        particles = H_funcs.V(particles).clip(-1, 1)
        H_funcs = H_funcs.to("cuda:0")
        particles_mcg_diff.append(particles[0])
    particles_mcg_diff = torch.concat(particles_mcg_diff, dim=0)
    return particles_mcg_diff


def run_dps(dps_config, alphas_cumprod, pipeline, model, sigma_y, N_MAX, D_OR, D_OBS, y_0, H_funcs, n_steps):
    batch_size = min(dps_config.N_total, N_MAX)
    N_batches = math.ceil(dps_config.N_total / batch_size)
    initial_particles = torch.randn(size=(N_batches, batch_size, *D_OR)).cpu()
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]

    try:
        sampler = create_sampler(sampler='ddim',
                                 steps=len(alphas_cumprod),
                                 noise_schedule=pipeline.scheduler.beta_schedule,
                                 model_mean_type='epsilon',
                                 model_var_type=pipeline.scheduler.variance_type,
                                 dynamic_threshold=False,
                                 clip_denoised=False,
                                 rescale_timesteps=True,
                                 timestep_respacing=f'ddim{n_steps}')

    except:
        from dps.guided_diffusion.gaussian_diffusion import get_sampler, space_timesteps

        sampler = get_sampler(name='ddim')(
            use_timesteps=space_timesteps(len(alphas_cumprod), f'ddim{n_steps}'),
            betas=betas.cpu(),
            model_mean_type='epsilon',
            model_var_type=pipeline.scheduler.variance_type,
            dynamic_threshold=False,
            clip_denoised=False,
            rescale_timesteps=True
        )
    particles_dps = []
    for j, batch_initial_particles in enumerate(initial_particles):
        particles_dps.append(sampler.p_sample_loop(model=EpsilonNetDDRM(unet=model),
                                                   x_start=batch_initial_particles.cuda(),
                                                   measurement=y_0[None, ...].repeat(batch_initial_particles.shape[0], 1, 1, 1),
                                                   measurement_cond_fn=PosteriorSampling(
                                                       operator=DPSHfuncOperator(H_func=H_funcs, dim=D_OBS),
                                                       noiser=get_noise('gaussian', sigma=sigma_y),
                                                       scale=dps_config.lr).conditioning,
                                                   record=False,
                                                   save_root=False).clip(-1, 1).cpu())
    return torch.concatenate(particles_dps, dim=0)


def run_ddrm(ddrm_cfg, timesteps, alphas_cumprod, model, y_0, H_funcs, N_MAX, D_OR, sigma_y):
    batch_size = min(ddrm_cfg.N_total, N_MAX)
    N_batches = math.ceil(ddrm_cfg.N_total / batch_size)
    ddrm_timesteps = timesteps.clone()
    ddrm_timesteps[-1] = ddrm_timesteps[-1] - 1
    initial_particles = torch.randn((N_batches, batch_size, *D_OR)).cpu()
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]

    particles_ddrm = []
    for j, batch_initial_particles in enumerate(initial_particles):
        particles_ddrm.append(efficient_generalized_steps(
            x=batch_initial_particles.cuda(),
            b=betas,
            seq=ddrm_timesteps.cpu(),
            model=EpsilonNetDDRM(unet=model),
            y_0=y_0[None, ...].cuda(),
            H_funcs=H_funcs,
            sigma_0=sigma_y,
            etaB=ddrm_cfg.etaB,
            etaA=ddrm_cfg.etaA,
            etaC=ddrm_cfg.etaC,
            classes=None,
            cls_fn=None)[0][-1].clip(-1, 1).cpu())
    particles_ddrm = torch.concat(particles_ddrm, dim=0)
    return particles_ddrm


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    full_path_images = os.path.join(cfg.save_folder,
                                    cfg.task.name,
                                    cfg.dataset.hf_model_tag.replace('-', '_').replace('/','_'),
                                    str(cfg.seed),
                                    'images')
    full_path_data = os.path.join(cfg.save_folder,
                                  cfg.task.name,
                                  cfg.dataset.hf_model_tag.replace('-', '_').replace('/','_'),
                                  str(cfg.seed),
                                  'data')
    Path(full_path_images).mkdir(parents=True, exist_ok=True)
    Path(full_path_data).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    # Loading HF model
    pipeline, x_origin, D_OR, D_FLAT = load_hf_model(cfg.dataset)
    fig = plot(x_origin)
    if cfg.plot:
        fig.show()
    if cfg.save_fig:
        fig.savefig(f'{full_path_images}/sample.pdf')
    plt.close(fig)

    H_funcs, y_0, y_0_origin, y_0_img, U_t_y_0, diag, coordinates_mask, D_OBS = load_operator(task_cfg=cfg.task,
                                                                                              D_OR=D_OR,
                                                                                              x_origin=x_origin)

    fig = plot(y_0_img)
    if cfg.plot:
        fig.show()
    if cfg.save_fig:
        fig.savefig(f'{full_path_images}/measure.pdf')
    plt.close(fig)


    #Diffusion stuff
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.cuda().clip(1e-6, 1)
    timesteps = torch.linspace(0, 999, cfg.diffusion.n_steps).long().cuda()
    eta = cfg.diffusion.eta

    model = pipeline.unet
    model = model.requires_grad_(False)
    model = model.eval()

    ## MCG_DIFF
    if cfg.mcg_diff:
        particles_mcg_diff = run_mcg_diff(
            mcg_diff_config=cfg.mcg_diff,
            n_max_gpu=cfg.dataset.N_MAX_GPU_MCG_DIFF,
            dim=D_FLAT,
            U_t_y_0=U_t_y_0,
            diag=diag,
            coordinates_mask=coordinates_mask,
            sigma_y=cfg.task.sigma_y,
            timesteps=timesteps,
            eta=eta,
            H_funcs=H_funcs,
            score_model=ScoreModel(net=torch.nn.DataParallel(EpsilonNetSVD(H_funcs, model, dim=D_OR).requires_grad_(False)),
                                   alphas_cumprod=alphas_cumprod,
                                   device='cuda:0'),
        )
        particles_mcg_diff = particles_mcg_diff.reshape(-1, *D_OR)

        furthest = find_furthest_particles_in_clound(particles_mcg_diff)
        for i, particle in enumerate(particles_mcg_diff[furthest]):
            fig = plot(particle)
            if cfg.plot:
                fig.show()
            if cfg.save_fig:
                fig.savefig(f'{full_path_images}/furthest_{i}_mcg_diff.pdf')
            plt.close(fig)
        if cfg.save_data:
            np.save(file=f'{full_path_data}/particles_mcg_diff.npy',
                    arr=particles_mcg_diff.cpu().numpy())

    if cfg.dps:
        particles_dps = run_dps(dps_config=cfg.dps,
                                alphas_cumprod=alphas_cumprod,
                                pipeline=pipeline,
                                model=model,
                                sigma_y=cfg.task.sigma_y,
                                N_MAX=cfg.dataset.N_MAX_GPU_DPS,
                                D_OR=D_OR,
                                y_0=y_0,
                                D_OBS=D_OBS,
                                H_funcs=H_funcs,
                                n_steps=cfg.diffusion.n_steps)

        for i, particle in enumerate(particles_dps[find_furthest_particles_in_clound(particles_dps)]):
            fig = plot(particle)
            if cfg.plot:
                fig.show()
            if cfg.save_fig:
                fig.savefig(f'{full_path_images}/furthest_{i}_dps.pdf')
            plt.close(fig)
        if cfg.save_data:
            np.save(file=f'{full_path_data}/particles_dps.npy',
                    arr=particles_dps.cpu().numpy())

    if cfg.ddrm:
        particles_ddrm = run_ddrm(
            ddrm_cfg=cfg.ddrm,
            timesteps=timesteps,
            alphas_cumprod=alphas_cumprod,
            model=model,
            sigma_y=cfg.task.sigma_y,
            N_MAX=cfg.dataset.N_MAX_GPU_DDRM,
            D_OR=D_OR,
            y_0=y_0,
            H_funcs=H_funcs,
        )
        for i, particle in enumerate(particles_ddrm[find_furthest_particles_in_clound(particles_ddrm)]):
            fig = plot(particle)
            if cfg.plot:
                fig.show()
            if cfg.save_fig:
                fig.savefig(f'{full_path_images}/furthest_{i}_ddrm.pdf')
            plt.close(fig)
        if cfg.save_data:
            np.save(file=f'{full_path_data}/particles_ddrm.npy',
                    arr=particles_ddrm.cpu().numpy())

    if cfg.save_data:
        np.save(file=f'{full_path_data}/noisy_obs.npy', arr=y_0.cpu().numpy())
        np.save(file=f'{full_path_data}/sample.npy', arr=x_origin.cpu().numpy())
        np.save(file=f'{full_path_data}/noiseless_obs.npy', arr=y_0_origin.cpu().numpy())


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

