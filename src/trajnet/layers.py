import torch
from torch import nn


class MixTrajectory(nn.Module):
    def __init__(self, dim_traj, len_traj, d_emb) -> None:
        super().__init__()

        self.dim_traj = dim_traj
        self.len_traj = len_traj
        self.d_emb = d_emb

        self.layer = nn.Linear(dim_traj * len_traj, d_emb * len_traj)

    def forward(self, batch_trajectories: torch.Tensor) -> torch.Tensor:
        """Mix time steps of trajectories.

        Parameters
        ----------
        batch_trajectories : Tensor, shape (batch_size, len_traj, dim_traj)

        Return
        ------
        out : Tensor, shape (batch_size, len_traj, d_emb)
        """
        out = self.layer(batch_trajectories.view(-1, self.dim_traj * self.len_traj))
        return out.view(-1, self.len_traj, self.d_emb)


class PositionalEncoder(nn.Module):
    """Positional encoder as described in [1].

    Account for the ordering of the elements in a sequence
    by adding an absolute positional-dependent vector to each element.
    Code adapted from `https://pytorch.org/tutorials/beginner/
    transformer_tutorial.html#define-the-model`_ and `https://github.com/tensorflow/
    models/blob/v2.15.0/official/vision/modeling/layers/nn_layers.py#L482-L647`_.

    Parameters
    ----------
    d_model : int
        The dimension of the elements of the sequence.
        For convenience, it should be an even number, otherwise code will breaks.

    dropout : float, default=0.1
        The probability in the dropout layer.

    max_len : int, default=5000
        The max-length of the sequence.

    dtype : torch.dtype, default=torch.float32
        The data type.

    References
    ----------
    ..[1] Vaswani, Ashish, et al. "Attention is all you need." NeurIPS 30 (2017).
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.max_len = max_len
        self.dtype = torch.float32 if dtype is None else dtype

        positions = torch.arange(max_len)
        frequencies = 10_000 ** (-torch.arange(0, d_model, 2) / d_model)
        grid_pos_freq = torch.outer(positions, frequencies)

        encodings = torch.zeros(size=(max_len, d_model), dtype=self.dtype)
        encodings[:, ::2] = torch.sin(grid_pos_freq)
        encodings[:, 1::2] = torch.cos(grid_pos_freq)

        self.dropout_layer = nn.Dropout(p=dropout)

        # make `encodings` part of the state of `PositionalEncoder`
        # use register_buffer as those encodings are not trainable
        self.register_buffer("encodings", encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``axis=-2``.

        Parameters
        ----------
        x : Tensor, shape (batch_size, n_elements, d_model) | (n_elements, d_model)
        """
        x_encodings = self.encodings[: x.size(-2)]
        return self.dropout_layer(x + x_encodings)
