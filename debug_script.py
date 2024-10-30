# %%
import torch

from py_source.sampling.repaint import repaint_svd
from py_source.sampling.epsilon_net import EpsilonNetSVD
from py_source.utils import load_epsilon_net, load_image, display_image


device = "cuda:0"
torch.set_default_device(device)


# load the image
img_path = "./material/ffhq_img/00018.png"
x_origin = load_image(img_path, device)

# load the degradation operator
path_operator = f"./material/degradation_operators/outpainting_half.pt"
degradation_operator = torch.load(path_operator, map_location=device)

# apply degradation operator
y = degradation_operator.H(x_origin[None])
y = y.squeeze(0)

# add noise
sigma = 0.01
y = y + sigma * torch.randn_like(y)

inverse_problem = (y, degradation_operator, sigma)

# %%
# load model with 500 diffusion steps
n_steps = 300
eps_net = load_epsilon_net("ffhq", n_steps, device)

eps_net_svd = EpsilonNetSVD(
    net=eps_net.net,
    alphas_cumprod=eps_net.alphas_cumprod,
    timesteps=eps_net.timesteps,
    H_func=degradation_operator,
    device=device,
)

# solve problem
initial_noise = torch.randn((1, 3, 256, 256), device=device)
reconstruction = repaint_svd(initial_noise, inverse_problem, eps_net_svd)

# %%
display_image(reconstruction[0])
# %%
reconstruction.shape
# %%
# plot results
import math
import matplotlib.pyplot as plt


# reshape y
n_channels = 3
n_pixel_per_channel = y.shape[0] // n_channels
hight = width = int(math.sqrt(n_pixel_per_channel))

y_reshaped = y.reshape(n_channels, hight, width)

# init figure
fig, axes = plt.subplots(1, 3)

images = (x_origin, y_reshaped, reconstruction[0])
titles = ("original", "degraded", "reconstruction")

# display figures
for ax, img, title in zip(axes, images, titles):
    display_image(img, ax)
    ax.set_title(title)

fig.tight_layout()

# %%
