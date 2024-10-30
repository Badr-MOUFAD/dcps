# Data taken from https://github.com/vita-epfl/trajnetplusplusdataset/tree/master/data/trajnet_original
# in the `crowds` folder.
# Original version of the data can be found in https://graphics.cs.ucy.ac.cy/portfolio in `Crowds Data`


from typing import Tuple

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from local_paths import LARGE_FILE_DIR


DATA_DIR = LARGE_FILE_DIR  / "trajectories/raw_data/"

DATA_MAPPING = {
    "ucy_student_1": DATA_DIR / "ucy" / "students_1.txt",
    "ucy_student_3": DATA_DIR / "ucy" / "students_3.txt",
}

DATA_TYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}

RANDOM_STATE = 1234


def load_data(
    name: str = "ucy_student_1",
    dtype: str = "float32",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load UCY Pedestrian trajectories datasets.

    The data has four columns where
        - Column 1: frame Id
        - Column 2: pedestrian Id
        - Column 3, 4: xy-positions of the pedestrian
    The size of each pedestrian trajectory is 20 and was recorded in 2.5 FPS.
    Refer to [1] for details about the preprocessing.

    Parameters
    ----------
    name : str, default 'eth_hotel'
        Name of the dataset. See the keys in ``DATA_MAPPING``.

    dtype : str, default 'float32'
        Data type of the the tensor.

    Return
    ------
    train, test: Tuple of tensors
        The proportions of train, test are (0.8, 0.2) of the data.

    References
    ----------
    ..[1] Karttikeya Mangalam, Yang An, Harshayu Girase, and Jitendra Malik.
        "From goals, way-points & paths to long term human trajectory forecasting",
        IEEE/CVF, 2021
    """
    data = np.loadtxt(DATA_MAPPING[name])

    # data comprises trajectories of lengths 20
    n_chunks, r = divmod(len(data), 20)
    assert r == 0

    # ensure train and test contains chunks of length 20
    q, r = divmod(int(n_chunks), 5)
    train_size, test_size = q * 4 * 20, (q + r) * 20

    # split to train, test
    # don't shuffle to ease slicing later
    train, test = train_test_split(
        data,
        train_size=train_size,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=False,
    )

    # convert to tensor
    train, test = [torch.tensor(item, dtype=DATA_TYPE[dtype]) for item in [train, test]]

    return train, test


if __name__ == "__main__":
    # check with a plot
    import matplotlib.pyplot as plt

    train, test = load_data("ucy_student_3")

    # plot all trajectories
    for idx in range(0, len(train) // 20, 5):
        traj = train[20 * idx : 20 * (idx + 1)]
        traj_x, traj_y = traj[:, -2], traj[:, -1]

        plt.plot(traj_x, traj_y)
        plt.scatter(traj_x[-1], traj_y[-1], marker="D")
    plt.show()
