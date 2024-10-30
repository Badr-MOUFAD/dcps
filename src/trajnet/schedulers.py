import torch


def linear_schedule(n_diffusion_steps: int, device: str):
    # diffusion scheduler as described in "Denoising diffusion probabilistic models"
    # https://arxiv.org/abs/2006.11239
    diffusion_rate = torch.linspace(1e-3, 1e-1, n_diffusion_steps, device=device)

    return torch.cumprod(1 - diffusion_rate, dim=0)


def cosine_schedule(n_diffusion_steps: int, device: str):
    # diffusion scheduler as described in
    # "Improved Denoising Diffusion Probabilistic Models"
    # https://arxiv.org/abs/2102.09672
    s = 0.008
    steps = torch.arange(n_diffusion_steps, device=device)

    return torch.cos((steps / n_diffusion_steps + s) / (1 + s) * torch.pi / 2) ** 2


def get_schedulers(diffusion_scheduler: torch.Tensor):
    mean_schedule = torch.sqrt(diffusion_scheduler)
    std_schedule = torch.sqrt(1 - diffusion_scheduler)

    diffusion_rate = 1 - diffusion_scheduler[1:] / diffusion_scheduler[:-1]
    diffusion_rate_0 = torch.tensor(
        [1 - diffusion_scheduler[0]], device=diffusion_scheduler.device
    )
    diffusion_rate = torch.hstack((diffusion_rate_0, diffusion_rate))

    return mean_schedule, std_schedule, diffusion_rate
