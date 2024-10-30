import sys
import os

import numpy as np
import torch
import yaml
from PIL import Image

from inverse_problem_with_diffusion_prior.ipwdp.diffusion import dict2namespace, Model
from inverse_problem_with_diffusion_prior.ipwdp.generative_models import ScoreModel
from inverse_problem_with_diffusion_prior.ipwdp.image_utils import save_jpg
from inverse_problem_with_diffusion_prior.ipwdp.optimal_particle_filter import particle_filter


class FlattenScoreModel(torch.nn.Module):

    def __init__(self, base_module, shape):
        super().__init__()
        self.base_module = base_module
        self.shape = shape

    def forward(self, x, t):
        return self.base_module(x.reshape(x.shape[0], *self.shape),
                                torch.tensor([t], device=x.device)).reshape(x.shape[0], -1)


if __name__ == '__main__':
    image_folder = sys.argv[1]
    image_id = sys.argv[2]
    destination_folder = sys.argv[3]
    n_particles = int(sys.argv[4])
    n_samples_per_gpu = int(sys.argv[5])
    n_samples = int(sys.argv[6])
    mask_name = str(sys.argv[7])
    original_noise_std = float(sys.argv[8])
    device = "cuda:0"
    print("Loading model")
    eta = 1
    n_steps = 250
    ckpt = 'models/celeba/celeba_hq.ckpt'
    with open('models/celeba/celeba_hq.yml', "r") as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)
    model = Model(new_config)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    model = model.requires_grad_(False)
    model = model.eval()
    betas = torch.linspace(new_config.diffusion.beta_start, new_config.diffusion.beta_end, new_config.diffusion.num_diffusion_timesteps)
    alphas = (1 - betas)
    shape = (3, 256, 256)
    base_model = torch.nn.DataParallel(FlattenScoreModel(base_module=model.eval(),
                                                         shape=(3, 256, 256)))
    base_model.to(device)
    score_model = ScoreModel(net=base_model,
                             alphas_cumprod=torch.cat((torch.ones((1,)), torch.cumprod(alphas, dim=0)), dim=0).to(device)[:-1],
                             device=device)

    var_observation = original_noise_std ** 2
    timesteps = torch.linspace(0, 999, n_steps).long()


    measure_path = f'{image_folder}/{image_id}_{mask_name}_{original_noise_std}.png'
    mask = torch.from_numpy(np.load(f'inp_masks/{mask_name}.npy'))
    if os.path.isfile(measure_path):
        images_with_square = torch.from_numpy(np.array(Image.open(measure_path).convert("RGB")))
        images_with_square = (images_with_square.permute(2, 0, 1) / 127.5 - 1).permute(1, 2, 0).flatten(0, 1)

    else:
        image_test = torch.from_numpy(
            np.array(Image.open(f'{image_folder}/{image_id}.png').convert("RGB"))
        )
        image_test = image_test.permute(2, 0, 1) / 127.5 - 1

        images_with_square = -torch.ones_like(image_test.permute(1, 2, 0).flatten(0, 1))
        images_with_square[mask.flatten() == 1, :] = image_test.permute(1, 2, 0).flatten(0, 1)[mask.flatten() == 1, :]
        images_with_square[mask.flatten() == 1, :] += torch.randn_like(images_with_square[mask.flatten() == 1, :])*original_noise_std

        save_jpg(images_with_square.permute(1, 0).reshape(shape), path=measure_path, format='png')

    init_obs = images_with_square[mask.flatten() == 1]
    diag = torch.ones(size=(int(mask.sum().item()),))
    coordinates_mask = mask.flatten()

    measurement = init_obs.T.flatten()

    diag = torch.stack((diag,)*3, dim=0).flatten()
    coordinate_mask = torch.stack((coordinates_mask,)*3, dim=0).flatten()
    dim = len(coordinate_mask)


    torch.cuda.empty_cache()
    destination =f'{destination_folder}/{mask_name}_n_steps_{n_steps}_std_{original_noise_std}'
    from inverse_problem_with_diffusion_prior.ipwdp.image_utils import display_sample

    fig = display_sample(images_with_square.permute(1, 0).reshape(3, 256, 256))
    fig.show()
    if not os.path.exists(destination):
        os.makedirs(destination)

    for i in range(n_samples):
        initial_particles = torch.randn(n_particles, dim).to(device)

        particles = particle_filter(
            initial_particles=initial_particles.cpu(),
            observation=measurement.cpu(),
            score_model=score_model,
            coordinates_mask=coordinate_mask.cpu(),
            likelihood_diagonal=diag.cpu(),
            var_observation=var_observation,
            timesteps=timesteps,
            eta=eta,
            n_samples_per_gpu_inference=n_samples_per_gpu,
            gaussian_var=0
        )

        particles = particles.clamp(-1, 1)

        save_jpg(sample=particles[10].reshape(shape),
                 path=f'{destination}/{i}.pdf')
        torch.cuda.empty_cache()