# MCGDIFF

This repository contains the code used for the paper entitled "Monte Carlo guided Diffusion for Bayesian Linear inverse problems".
The main component is the "package" ipwdp (for inverse problems with diffusion prior).
It should be used as a package. A way of doing so is by defining the root of the repository in the PYTHONPATH variable.
This can be done by the following command:
`
export PYTHONPATH=.:$PYTHONPATH
`

The requirements.txt file show the minimal required packages to run the files.

## GMM:

To run the Gaussian Mixture model experiments, you need to run the `ggm.py` file from root. It will create
pdf of images in an `images/gmm` folder that has to be made before running the script. It will also create a csv
reporting the sliced wasserstein for each run in a data folder that has to be made. In order to run this repo,
one also needs to clone the DDRM repository to run in the comparison. `https://github.com/bahjat-kawar/ddrm`

## Inpainting

To run the image inpainting experiment, you need to run the `image_inpainting.py` script.

It takes the following arguments (in order)
* image_folder: The path to the folder containing the images to be used for the inpainting
* image_id: the name of the image file to be used (without the .png suffix)
* destination_folder: The path of the file to save the experiments
* n_particles: The number of total particles to be used at each approximation.
* n_samples_per_gpu: Number of particles each gpu can handle. To be adjusted according to your hardware.
* n_samples: Number of times to run the inference process.
* mask_name: The name of one of the masks available at the folder /inp_masks.
* original_noise_std: Float indicating the measurement noise std.

The code makes use of nn.DataParallel from torch. If you want to run it on a subset of devices, one should set
the environment variable `CUDA_VISIBLE_DEVICES` to only the devices one wants to use.


## Debluring and Super Resolution

For both Debluring and Super Resolution, the script needed to run the simulations is `run_hf.py`.
The configs folder and `config.yaml` dictates the experiment run by the script.