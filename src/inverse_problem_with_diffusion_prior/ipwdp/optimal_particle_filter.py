from typing import Tuple, List

import numpy as np
import torch
import tqdm
from torch.distributions import Categorical

from inverse_problem_with_diffusion_prior.ipwdp.generative_models import ScoreModel, generate_coefficients_ddim
from inverse_problem_with_diffusion_prior.ipwdp.inverse_problems_utils import get_taus_from_singular_values


def predict(score_model: ScoreModel,
            particles: torch.Tensor,
            t: float,
            t_prev: float,
            eta: float,
            n_samples_per_gpu: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise, coeff_sample, coeff_score = generate_coefficients_ddim(
        alphas_cumprod=score_model.alphas_cumprod.to(particles.device),
        time_step=t,
        prev_time_step=t_prev,
        eta=eta
    )
    if hasattr(score_model.net, 'device_ids'):
        batch_size = n_samples_per_gpu * len(score_model.net.device_ids)
        epsilon_predicted = []
        n_batches = particles.shape[0] // batch_size + int(particles.shape[0] % batch_size > 0)
        for batch_idx in range(n_batches):
            epsilon_predicted.append(score_model.net(particles[batch_size*batch_idx:(batch_idx+1)*batch_size], t).cpu())
        epsilon_predicted = torch.cat(epsilon_predicted, dim=0).to(particles.device)
    else:
        epsilon_predicted = score_model.net(particles, t).to(particles.device)
    mean = coeff_sample * particles + coeff_score * epsilon_predicted.to(particles.device)

    return mean, noise, epsilon_predicted


def particle_filter(initial_particles: torch.Tensor,
                    observation: torch.Tensor,
                    score_model: ScoreModel,
                    coordinates_mask: torch.Tensor,
                    likelihood_diagonal: torch.Tensor,
                    var_observation: torch.Tensor,
                    timesteps: List[int],
                    eta: float = 1,
                    n_samples_per_gpu_inference: int = 16,
                    gaussian_var: float = 1e-2):
    all_parts = []
    n_particles, dim = initial_particles.shape
    alphas_cumprod = score_model.alphas_cumprod.to(initial_particles.device)

    log_weights = torch.zeros((n_particles,), device=initial_particles.device)
    particles = initial_particles
    #all_particles = [particles]
    taus, taus_indices = get_taus_from_singular_values(alphas_cumprod=alphas_cumprod,
                                                       timesteps=timesteps,
                                                       singular_values=likelihood_diagonal,
                                                       var=var_observation)

    coordinates_in_state = torch.where(coordinates_mask == 1)[0]
    always_free_coordinates = torch.where(coordinates_mask == 0)[0]
    rescaled_observation = (alphas_cumprod[taus]**.5)*observation / likelihood_diagonal
    pbar = tqdm.tqdm(enumerate(zip(timesteps.tolist()[1:][::-1],
                                   timesteps.tolist()[:-1][::-1])),
                     desc='Particle Filter')
    for i, (t, t_prev) in pbar:
        # first create the prediction for each particle of the previous particle cloud
        predicted_mean, predicted_noise, eps = predict(score_model=score_model,
                                                       particles=particles,
                                                       t=t,
                                                       t_prev=t_prev,
                                                       eta=eta,
                                                       n_samples_per_gpu=n_samples_per_gpu_inference)

        ancestors = Categorical(logits=log_weights).sample((n_particles,))
        ptg_anc = len(set(ancestors.tolist())) / len(ancestors)
        pbar.set_postfix({'ptg Ancestors': ptg_anc}, refresh=False)

        new_particles = torch.empty_like(particles)
        new_log_weights = torch.zeros_like(log_weights)
        coordinates_to_filter = coordinates_in_state[taus < t_prev]
        exactly_observed_coordinates = coordinates_in_state[taus == t_prev]
        free_coordinates = torch.cat((coordinates_in_state[taus > t_prev], always_free_coordinates), dim=0)
        if len(coordinates_to_filter) > 0:
            # First we rescale the alphas, by dividing by the taus of the observed variable
            alpha_t_prev = alphas_cumprod[t_prev] / alphas_cumprod[taus[taus < t_prev]]

            diffused_observation = rescaled_observation[taus < t_prev] * (alpha_t_prev ** .5)
            observation_std = ((1 - (1-gaussian_var)*alpha_t_prev) ** .5)
            top_predicted_mean = predicted_mean[ancestors, :][:, coordinates_to_filter]

            #BISHOP 2.115 adn 2.116
            posterior_precision = (1 / (predicted_noise ** 2)) + (1 / (observation_std ** 2))
            posterior_mean = (1 / posterior_precision)[None, :] * (
                    diffused_observation[None, :] / (observation_std ** 2) + (
                    top_predicted_mean/ (predicted_noise ** 2)))

            noise_top = torch.randn_like(posterior_mean)
            top_samples = posterior_mean + noise_top*((1/posterior_precision)**.5)

            log_integration_constant = -.5 * torch.linalg.norm((top_predicted_mean - diffused_observation[None, :]) / ((predicted_noise ** 2 + observation_std**2)[None, :]**.5), dim=-1)**2

            alpha_t = alphas_cumprod[t] / alphas_cumprod[taus[taus < t_prev]]
            top_previous_particles = particles[ancestors][:, coordinates_to_filter].clone()
            previous_residue = (top_previous_particles - rescaled_observation[None, taus < t_prev] * (alpha_t[None, :] **.5)) / ((1 - (1 - gaussian_var)*alpha_t[None, :])**.5)
            log_forward_previous_likelihood = -.5 * torch.linalg.norm(previous_residue, dim=-1)**2

            new_log_weights += log_integration_constant - log_forward_previous_likelihood
            new_particles[:, coordinates_to_filter] = top_samples

        if len(exactly_observed_coordinates) > 0:

            if t_prev == 0:
                # Do not use proposal kernel and simply do a bootstrap update
                alpha_t = alphas_cumprod[t] / alphas_cumprod[taus[taus == t_prev]]
                top_previous_particles = particles[ancestors][:, exactly_observed_coordinates].clone()
                previous_residue = (top_previous_particles - rescaled_observation[None, taus == t_prev] * (
                        alpha_t[None, :] ** .5)) / ((1 - (1-gaussian_var)*alpha_t[None, :]) ** .5)
                log_forward_previous_likelihood = -.5 * torch.linalg.norm(previous_residue, dim=-1) ** 2

                top_predicted_mean = predicted_mean[ancestors, :][:, exactly_observed_coordinates]
                log_backward_transition = -.5 * torch.linalg.norm(
                    (top_predicted_mean - rescaled_observation[None, taus == t_prev]) / predicted_noise, dim=-1) ** 2
                log_weights += log_backward_transition - log_forward_previous_likelihood
                new_particles[:, exactly_observed_coordinates] = rescaled_observation[None, taus==t_prev]
            else:
                diffused_observation = rescaled_observation[taus == t_prev].clone()
                observation_std = gaussian_var**.5
                top_predicted_mean = predicted_mean[ancestors, :][:, exactly_observed_coordinates]

                posterior_precision = (1 / (predicted_noise ** 2)) + (1 / (observation_std ** 2))
                posterior_mean = (1 / posterior_precision) * (
                        (diffused_observation / (observation_std ** 2))[None, :] + (
                        top_predicted_mean/ (predicted_noise ** 2)))

                noise_top = torch.randn_like(posterior_mean)
                top_samples = posterior_mean + noise_top*((1/posterior_precision)**.5)

                log_integration_constant = -.5 * torch.linalg.norm((top_predicted_mean - diffused_observation[None, :]) / ((predicted_noise ** 2 + observation_std**2)**.5), dim=-1)**2

                alpha_t = alphas_cumprod[t] / alphas_cumprod[taus[taus == t_prev]]
                top_previous_particles = particles[ancestors][:, exactly_observed_coordinates].clone()
                previous_residue = (top_previous_particles - rescaled_observation[None, taus == t_prev] * (alpha_t[None, :] **.5)) / ((1 - (1 - gaussian_var)*alpha_t[None, :])**.5)
                log_forward_previous_likelihood = -.5 * torch.linalg.norm(previous_residue, dim=-1)**2

                new_log_weights += log_integration_constant - log_forward_previous_likelihood
                new_particles[:, exactly_observed_coordinates] = top_samples

        if len(free_coordinates) > 0:
            noise_bottom = torch.randn_like(predicted_mean[:, free_coordinates])
            bottom_samples = predicted_mean[ancestors, :][:, free_coordinates] + noise_bottom * predicted_noise
            new_particles[:, free_coordinates] = bottom_samples

        log_weights = new_log_weights.clone()
        all_parts.append(new_particles.clone())
        particles = new_particles.clone()

    # import matplotlib.pyplot as plt
    # confidence_intervals = torch.stack([torch.stack((torch.min(p, axis=0)[0], torch.max(p, axis=0)[0]), dim=0) for p in all_parts[::-1]], dim=0)
    # diffused_observation = (alphas_cumprod[timesteps[:-1], None] ** .5) * (observation / likelihood_diagonal)[None, :]
    # for coord in (np.linspace(0, (len(likelihood_diagonal) - 1)**.5, 10)**2).astype(np.int):
    #
    #     plt.fill_between(x=timesteps[:-1],
    #                      y1=confidence_intervals[:, 0, coord],
    #                      y2=confidence_intervals[:, 1, coord],
    #                      color='red')
    #     plt.axvline(taus[coord])
    #     plt.plot(timesteps, (alphas_cumprod[timesteps]**.5)*(observation / likelihood_diagonal)[coord])
    #     plt.xscale('log')
    #     plt.show()
    # is_inside_before_tau = [all((confidence_intervals[timesteps[:-1] > taus[c], 0, c] < diffused_observation[timesteps[:-1] > taus[c], c]) &
    #                         (confidence_intervals[timesteps[:-1] > taus[c], 1, c] > diffused_observation[timesteps[:-1] > taus[c], c]))
    #                         for c in range(len(likelihood_diagonal))]
    # is_inside_after_tau = [all((confidence_intervals[timesteps[:-1] <= taus[c], 0, c] < diffused_observation[timesteps[:-1] <= taus[c], c]) &
    #                         (confidence_intervals[timesteps[:-1] <= taus[c], 1, c] > diffused_observation[timesteps[:-1] <= taus[c], c]))
    #                         for c in range(len(likelihood_diagonal))]
    # print(np.mean(is_inside_before_tau), np.mean(is_inside_after_tau))
    return particles#, np.mean(is_inside_before_tau), np.mean(is_inside_after_tau)#, all_particles
