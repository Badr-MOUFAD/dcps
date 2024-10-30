# %%
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from trajnet.datasets.pedestrians import load_data
from trajnet.models import DDPM
from trajnet.schedulers import cosine_schedule
from trajnet.denoisers import EpsilonNet

import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(1234)

# %%
# setups

dim_traj = 2
len_traj = 20

n_epochs = 1000
batch_size = 145  # ensure batch_size ~10 times the data size
batch_size_test = 141  # total size of test

# optimizer and scheduler
max_lr = 1e-3
min_lr = 1e-5
T_period = n_epochs

n_diffusion_steps = 1000
d_feedforward = 2048
d_intermediate = 1024
d_emb = 512
dropout = 0.0

device = "cuda:1"

# %%
# init
train_1, test_1 = load_data("ucy_student_1")
train_3, test_3 = load_data("ucy_student_3")

data_train = torch.vstack((train_1, test_1, train_3))
data_test = torch.vstack((test_3,))

data_train = data_train - data_train.mean(dim=0)
data_train = data_train / data_train.std(dim=0)

data_test = data_test - data_test.mean(dim=0)
data_test = data_test / data_test.std(dim=0)

# keep xy and reshape
data_train = data_train[:, 2:].reshape(-1, len_traj, dim_traj)[1:]
data_test = data_test[:, 2:].reshape(-1, len_traj, dim_traj)

# %%

train_loader = DataLoader(data_train, batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size_test, shuffle=True)

# noise scheduler
alpha_cum_prod = cosine_schedule(n_diffusion_steps, device)

# model
model = DDPM(
    EpsilonNet(
        dim_traj,
        len_traj,
        alpha_cum_prod,
        d_emb=d_emb,
        d_feedforward=d_feedforward,
        d_intermediate=d_intermediate,
        dropout=dropout,
    ),
    device=device,
)

optimizer = torch.optim.Adam(model.parameters(), max_lr)

lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_period, eta_min=min_lr)

train_losses, test_losses, post_losses = [], [], []

# %%
# sanity check of diffusion process
model._plot_forward_diffusion(
    data_train[:2000],
    diffusion_steps=[0, n_diffusion_steps // 2, n_diffusion_steps - 1],
)

# %%
# train/test loop
import time

start = time.perf_counter()

for epoch in (tbar := tqdm.trange(n_epochs)):

    #####
    # train pass
    #####
    for batch_train in train_loader:
        model.train()
        optimizer.zero_grad()

        loss_per_sample = model.compute_batch_loss(batch_train)
        loss_per_sample.backward()

        optimizer.step()

    train_losses.append(loss_per_sample.cpu().item())

    #####
    # test pass
    #####

    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        loss_per_sample = model.compute_batch_loss(batch)
        test_losses.append(loss_per_sample.cpu().item())

        batch = next(iter(train_loader))
        loss_per_sample = model.compute_batch_loss(batch)
        post_losses.append(loss_per_sample.cpu().item())

    lr_scheduler.step()

    tbar.set_description(
        f"train loss = {train_losses[-1]:.8f}, test loss = {test_losses[-1]:.8f}"
    )

end_time = time.perf_counter()

# %%
# plot train/test losses

plt.plot(train_losses, label="train")
plt.plot(post_losses, label="post train")
plt.plot(test_losses, label="test")

plt.xlabel("epochs")
plt.ylabel("Loss per sample")

plt.legend()
plt.show()

# %%
# generate samples

model.eval()
model.requires_grad_(False)

n_new_trajectories = 30

new_trajectories = model.sample(n_new_trajectories)
new_trajectories = new_trajectories.cpu()

# %%
color = "#1f77b4"
fig, ax = plt.subplots()

color = {"generated": "#1f77b4", "real": "#ff7f0e"}


for idx, traj in enumerate(new_trajectories[:20]):
    traj_x, traj_y = traj[:, 0], traj[:, 1]

    ax.plot(
        traj_x,
        traj_y,
        marker=".",
        color=color["generated"],
    )
    ax.scatter(
        traj_x[-1],
        traj_y[-1],
        marker="D",
        color=color["generated"],
    )

# small hack to get one legend item
traj_x, traj_y = new_trajectories[0, :, 0], new_trajectories[0, :, 1]
ax.scatter(
    traj_x[-1], traj_y[-1], marker="D", color=color["generated"], label="generated"
)

indices = torch.randint(0, len(data_train), size=(30,))

for traj in data_train[indices]:
    traj_x, traj_y = traj[:, 0], traj[:, 1]

    ax.plot(
        traj_x,
        traj_y,
        marker=".",
        color=color["real"],
        alpha=0.5,
    )
    ax.scatter(
        traj_x[-1],
        traj_y[-1],
        marker="D",
        color=color["real"],
        alpha=0.5,
    )

# small hack to get one legend item
traj_x, traj_y = data_train[0, :, 0], data_train[0, :, 1]
ax.scatter(traj_x[-1], traj_y[-1], marker="D", color=color["real"], label="real")


ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title(f"Trajectories of length {len_traj}")
ax.set_aspect("equal", adjustable="box")
plt.legend()

# %%
# path = "../checkpoints/ucy_len_20_n_diff_steps_1000__.pt"

# torch.save(model.denoiser, path)
