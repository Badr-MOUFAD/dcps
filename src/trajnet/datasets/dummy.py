import torch
import matplotlib.pyplot as plt


def dummy_traj(
    dim_traj: int = 2,
    len_traj: int = 10,
    n_traj: int = 100,
    std_noise: float = 0.1,
    random_state: int = 1234,
) -> torch.Tensor:
    """Generate hyperbolic trajectories.

    A trajectory is of the form

        traj[t]= (t ** 2 * a*z_2 + t * b*z_1 + c*z_0) + std * noise[t]

    where ``noise`` is a standard Gaussian and
        - ``t`` is time and ranges between ``[0, len_traj]``
        - ``z_1, z_2, z_3`` are 2D standard Gaussian
        - a, b, c are coeficients to control the shape of trajectory

    Parameters
    ----------
    len_traj : int, default=10
        The length of trajectory.

    dim_traj : int, default=2
        The dimension of trajectory.

    n_traj : int, default=100
        The number of trajectories to generate.

    std_noise : float, default=0.1
        The standard deviation of the noise added to each point of the trajectory.

    random_state: int, default=1234
        Seed to fix randomness

    Return
    ------
    trajectories : Tensor, shape (n_traj, len_traj, dim_traj)
    """
    torch.manual_seed(random_state)

    drift_coefs = torch.tensor([0, 1, -1 / (2 * len_traj)])
    n_drift_coefs = len(drift_coefs)

    # coefs of all trajectories
    traj_coefs = drift_coefs[None, :, None] * torch.randn(
        (n_traj, n_drift_coefs, dim_traj)
    )

    # grid of time
    time = torch.arange(len_traj)
    grid_time = time[:, None] ** torch.arange(n_drift_coefs)
    grid_time = grid_time.type(torch.float)

    # b: batch, c: coefs, d; dim_traj, t: len_traj
    trajectories = torch.einsum("bcd,tc->btd", traj_coefs, grid_time)
    trajectories = trajectories + std_noise * torch.randn_like(trajectories)

    # standardize
    trajectories = trajectories - trajectories.mean(dim=(0, 1))
    trajectories = trajectories / trajectories.std(dim=(0, 1))

    return trajectories


if __name__ == "__main__":
    trajectories = dummy_traj(n_traj=20, len_traj=50, std_noise=0.2)

    color = "#1f77b4"
    fig, ax = plt.subplots()

    for traj in trajectories:
        traj_x, traj_y = traj[:, 0], traj[:, 1]

        ax.plot(traj_x, traj_y, marker=".", color=color)
        ax.scatter(traj_x[-1], traj_y[-1], marker="D", color=color)

        ax.set_aspect("equal", adjustable="box")

    plt.show()
