import torch
import tqdm

from inverse_problem_with_diffusion_prior.ipwdp.generative_models import ScoreModel
from inverse_problem_with_diffusion_prior.ipwdp.inverse_problems_utils import get_taus_from_singular_values
from inverse_problem_with_diffusion_prior.ipwdp.optimal_particle_filter import predict


def smc_diff(initial_particles: torch.Tensor,
             observation: torch.Tensor,
             score_model: ScoreModel,
             likelihood_diagonal: torch.Tensor,
             coordinates_mask: torch.Tensor,
             var_observation: torch.Tensor,
             timesteps: torch.Tensor,
             eta: float = 1,
             n_samples_per_gpu: int = 128):
    n_particles = initial_particles.shape[0]

    particle_cloud = torch.empty((len(timesteps), n_particles, initial_particles.shape[-1]))
    log_weights = torch.empty((len(timesteps), n_particles))
    ancestors = torch.empty((len(timesteps), n_particles))

    alphas_cumprod = score_model.alphas_cumprod.cpu()
    taus, taus_indices = get_taus_from_singular_values(alphas_cumprod, timesteps, likelihood_diagonal, var_observation)

    taus = timesteps[taus_indices]
    diffused_motif = torch.zeros((len(timesteps), len(observation)))
    diffused_motif[taus_indices] = observation[None, :]*(alphas_cumprod[timesteps[taus_indices]]**.5) / likelihood_diagonal
    for i, tau_ind_uniq in enumerate(torch.unique(taus_indices)):
        for j, t in enumerate(timesteps[tau_ind_uniq + 1:]):
            index = j + tau_ind_uniq
            beta = 1 - alphas_cumprod[t] / alphas_cumprod[timesteps[index]]
            diffused_motif[index + 1, taus_indices == tau_ind_uniq] = ((1 - beta)**.5) * diffused_motif[index, taus_indices == tau_ind_uniq] + (beta**.5) * torch.randn_like(diffused_motif[index, taus_indices == tau_ind_uniq])

    observed_coordinates = torch.where(coordinates_mask == 1)[0]
    never_observed_coordinates = torch.where(coordinates_mask == 0)[0]
    free_coordinates = never_observed_coordinates.clone()
    current_particles = initial_particles

    n = len(observed_coordinates) + len(never_observed_coordinates)

    for i, (t, t_prev) in tqdm.tqdm(enumerate(zip(timesteps.tolist()[1:][::-1],
                                                  timesteps.tolist()[:-1][::-1])),
                                    desc='SMC diff'):
        rev_index = len(timesteps) - 2 - i
        if (t_prev >= taus).any():
            recomposed_full_data = torch.zeros((n_particles, n))
            recomposed_full_data[:, observed_coordinates] = diffused_motif[None, rev_index+1].repeat(n_particles, 1)
            recomposed_full_data[:, free_coordinates] = current_particles[:, free_coordinates]
            predicted_mean, predicted_noise, eps = predict(score_model=score_model,
                                                      particles=recomposed_full_data,
                                                      t=t,
                                                      t_prev=t_prev,
                                                      eta=eta,
                                                      n_samples_per_gpu=n_samples_per_gpu)

            next_motif_coordinates = observed_coordinates[taus <= t_prev]
            just_seen_coordinates = observed_coordinates[taus == t_prev]
            free_coordinates = torch.cat((observed_coordinates[taus > t_prev], just_seen_coordinates, never_observed_coordinates))
            current_lw = -.5 * (torch.linalg.norm(diffused_motif[rev_index, taus <= t_prev] - predicted_mean[:, next_motif_coordinates], dim=-1)**2)

            current_ancestors = torch.distributions.Categorical(logits=current_lw).sample((n_particles,))
            noise = torch.randn_like(predicted_mean)
            current_particles = predicted_mean[current_ancestors, :] + predicted_noise*noise
            current_particles[:, just_seen_coordinates] = diffused_motif[rev_index, taus==t_prev][None, :].repeat(n_particles, 1)
            particle_cloud[i] = recomposed_full_data[current_ancestors]
            ancestors[i] = current_ancestors
            log_weights[i] = current_lw[current_ancestors]
        else:
            current_lw = torch.ones((n_particles,))
            current_ancestors = torch.arange(n_particles)
            particle_cloud[i] = current_particles
            ancestors[i] = current_ancestors
            log_weights[i] = current_lw[current_ancestors]
            predicted_mean, predicted_noise, eps = predict(score_model=score_model,
                                                      particles=current_particles,
                                                      t=t,
                                                      t_prev=t_prev,
                                                      eta=eta,
                                                      n_samples_per_gpu=n_samples_per_gpu)
            current_particles = predicted_mean + torch.randn_like(predicted_mean)*predicted_noise
    i = i + 1
    particle_cloud[i] = current_particles
    ancestors[i] = current_ancestors
    log_weights[i] = current_lw[current_ancestors]

    return current_particles, particle_cloud





