import torch
from torch import nn

from trajnet.schedulers import get_schedulers
from trajnet.layers import PositionalEncoder, MixTrajectory


class EpsilonNet(nn.Module):

    def __init__(
        self,
        dim_traj: int,
        len_traj: int,
        alpha_cum_prod: torch.Tensor,
        d_emb: int = 16,
        d_feedforward: int = 256,
        d_intermediate: int = 64,
        dropout=0.1,
    ):
        super().__init__()

        self.dim_traj = dim_traj
        self.len_traj = len_traj

        self.d_emb = d_emb
        self.d_feedforward = d_feedforward
        self.d_intermediate = d_intermediate

        self.alpha_cum_prod = alpha_cum_prod
        self.n_diffusion_steps = len(alpha_cum_prod)
        self.mean_schedule, self.std_schedule, self.diffusion_rate = get_schedulers(
            alpha_cum_prod
        )

        # embed input and diffusion time
        self.encoder_block = MixTrajectory(dim_traj, len_traj, d_emb)
        self.position_encoder = PositionalEncoder(
            d_model=d_emb * len_traj, max_len=self.n_diffusion_steps, dropout=dropout
        )

        # capture temporal dependence
        self.transformer = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            PositionalEncoder(d_model=d_emb, max_len=len_traj),
            nn.TransformerEncoderLayer(
                d_model=d_emb,
                dim_feedforward=d_feedforward,
                batch_first=True,
                nhead=4,
                dropout=dropout,
            ),
        )

        # predict new trajectory
        self.pred_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_emb, d_intermediate),
            nn.ReLU(),
            nn.Linear(d_intermediate, d_intermediate),
            nn.ReLU(),
            nn.Linear(d_intermediate, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, dim_traj),
        )

    def predict_noise(
        self, batch_trajectories: torch.Tensor, t: int | torch.Tensor
    ) -> torch.Tensor:
        pos_encodings = self.position_encoder.encodings.view(
            -1, self.len_traj, self.d_emb
        )

        # encode
        encoded = self.encoder_block.forward(batch_trajectories)
        encoded = encoded + pos_encodings[t]

        # transform then predict
        transformed = self.transformer.forward(encoded)
        pred = self.pred_block.forward(transformed)

        return pred

    def denoise(self, batch_noisy_traj: torch.Tensor, t: int) -> torch.Tensor:
        # TODO handle case where t is a tensor
        repeated_time = torch.full(
            fill_value=t,
            size=(batch_noisy_traj.shape[0],),
            device=batch_noisy_traj.device,
        )

        pred_noise = self.predict_noise(batch_noisy_traj, repeated_time)
        scale_noise = self.diffusion_rate[t] / self.std_schedule[t]

        return (batch_noisy_traj - scale_noise * pred_noise) / torch.sqrt(
            1 - self.diffusion_rate[t]
        )

    def predict_trajectory(
        self,
        batch_trajectories: torch.Tensor,
        t: int | torch.Tensor,
    ) -> torch.Tensor:
        """Expected value of X_0 Knowing X_t"""
        pred_noise = self.predict_noise(batch_trajectories, t)

        std_schedule_t = self.std_schedule[t]
        mean_schedule_t = self.mean_schedule[t]
        pred = (batch_trajectories - std_schedule_t * pred_noise) / mean_schedule_t

        return pred

    def forward(self, batch_trajectories, t):
        # proxy of `predict_noise` method for compatibility with torch
        return self.predict_noise(batch_trajectories, t)
