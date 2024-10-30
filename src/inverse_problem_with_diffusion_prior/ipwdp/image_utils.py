from typing import Tuple

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def recompose_image_from_perpendicular_decomposition(x_perp: torch.Tensor,
                                                     x_parallel: torch.Tensor,
                                                     projection_perp: torch.Tensor,
                                                     projection_parallel: torch.Tensor,
                                                     image_dimensions: Tuple[int, int]) -> torch.Tensor:
    '''

    :param x_perp: Tensor of shape (batch_size, dim_perp, 3)
    :param x_parallel: Tensor of shape (batch_size, dim_parallel, 3)
    :param projection_perp: Tensor of shape (dim_perp, dim_image)
    :param projection_parallel: Tensor of shape (dim_paralle, dim_image)
    :param image_dimensions: Tuple to reshape
    :return: Image (batch_size, n_channels, image_dimensions[0], image_dimensions[1])
    '''
    dim_perp, dim_image = projection_perp.shape
    batch_size, _, n_channels = x_perp.shape
    image_perp = (projection_perp.T @ x_perp.permute(1, 0, 2).flatten(1, 2)).reshape(dim_image, batch_size, n_channels)
    image_parallel = (projection_parallel.T @ x_parallel.permute(1, 0, 2).flatten(1, 2)).reshape(dim_image, batch_size, n_channels)
    return (image_perp + image_parallel).permute(1, 2, 0).reshape(batch_size, n_channels, *image_dimensions)


def display_sample(sample):
    image_processed = sample.cpu().permute(1, 2, 0)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_pil)
    #.title(f"Image at step {i}")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def save_jpg(sample, path, format='PDF'):
    if sample.shape[0] == 3:
        image_processed = sample.cpu().permute(1, 2, 0)
    else:
        image_processed = sample[0].cpu()
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed, mode='L' if len(sample.shape)==2 else None)
    image_pil.save(fp=path, format=format)
    return None