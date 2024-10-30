# Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors


## Code installation

Install the code in editable mode

```bash
pip install -e .
```

This command will also download the code dependencies.
Further details about the code can be found in ``pyproject.toml``.

For convenience, the code of these repositories were moved inside ``src`` folder

- https://github.com/gabrielvc/mcg_diff
- https://github.com/bahjat-kawar/ddrm
- https://github.com/openai/guided-diffusion
- https://github.com/NVlabs/RED-diff
- https://github.com/mlomnitz/DiffJPEG

to avoid installation conflicts.


## Large files

The models checkpoints, datasets were ignored as they contain large files.
Make sure to create a folder ``large_files`` and download the right files and folders.

To avoid path conflict, ensure to insert in ``src/local_paths.py`` script

- the absolute path of the repository
- the path of the folder ``large_files``

and update the ``model_path`` in the configuration files ``ffhq_model.yaml`` and ``imagenet_model.yaml``.

The ``large_files`` folder have the following structure.
Make sure to preserve it.

```
  large_files/
  ├── ddm-inv-problems/
  ├──── ffhq/
  |    └── validation_set/
  |       └── im1.png
  |       └── ...
  |    └── ffhq_mt.pt
  ├──── imagenet/
  |    └── validation_set/
  |       └── im1.png
  |       └── ...
  ├──── masks_img256/
  |    └── inpainting_middle.pt
  |    └── ...
  |—— trajectories/
  |    └── raw_data/
  |       └── ucy/
  |          └── students_1.txt
  |          └── students_3.txt
  |    └── checkpoints/
  |       └── ucy_len_20_n_diff_steps_1000.pt
```

## Downloading checkpoints

- [Imagnet](https://github.com/openai/guided-diffusion)
- [FFHQ](https://github.com/DPS2022/diffusion-posterior-sampling)
- [Trajectories](https://drive.google.com/drive/folders/1gZb-kMX6TPuci7moDcwIMD15gfQLem7l?usp=share_link)
