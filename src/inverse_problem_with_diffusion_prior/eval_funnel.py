import os
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import pandas as pd
import sys


def sliced_wasserstein(dist_1, dist_2, n_slices=100):
    projections = np.random.normal(size=(n_slices, dist_1.shape[1]))
    projections = projections / np.linalg.norm(projections, axis=-1)[:, None]
    dist_1_projected = (projections @ dist_1.T)
    dist_2_projected = (projections @ dist_2.T)
    return np.mean([wasserstein_distance(u_values=d1,
                                         v_values=d2) for d1, d2 in zip(dist_1_projected, dist_2_projected)])


def make_table(metric_data):
    tab = pd.DataFrame.from_records(metric_data).groupby(['D_X',
                                                          'D_Y',
                                                          'N_components',
                                                          'n_ddim_steps',
                                                          'alg']).dist_value.agg(
        dist_value=lambda x: f'{np.mean(x):.2f} ± {(1.96*np.std(x) / np.sqrt(x.shape[0])):.2f}',
        dist_num=np.mean
    )
    tab = tab.reset_index()
    tab = tab.pivot(columns='alg', values=['dist_value', 'dist_num'], index=['D_X', 'D_Y', 'N_components',
                                                                             'n_ddim_steps'])
    tab.columns = tab.columns.map(lambda x: f'{x[1]}_num' if 'num' in x[0] else x[1])
    return tab.reset_index()


if __name__ == '__main__':
    folder = sys.argv[1]
    component = 'funnel'
    #component = 'gaussian'
    color_prior = '#492372'
    color_diffusion = '#ecee10'
    color_posterior = '#ade2e6' #a2c4c9'
    color_algorithm = '#ff7878'
    metric_data = []
    rnvp_losses = {}
    sw_priors = []
    for file in os.listdir(f'{folder}/{component}_mm'):
        if 'npz' in file:
            split = file.replace('.npz', '').split('_')
            if len(split) == 5:
                D_X, D_Y, operator_dim, N_components, seed = split
                n_steps = 100
            elif len(split) == 6:
                D_X, D_Y, operator_dim, N_components, seed, n_steps = split
            else:
                raise ValueError

            if int(N_components) == 20:
                data = np.load(f'{folder}/{component}_mm/{file}')
                try:
                    if (int(D_X), int(D_Y)) in rnvp_losses:
                        rnvp_losses[(int(D_X), int(D_Y))].append(data["loss_RNVP"])
                    else:
                        rnvp_losses[(int(D_X), int(D_Y))] = [data["loss_RNVP"]]
                except:
                    print(f'{folder}/{component}_mm/{file}')
                if component == 'funnel':
                    prior_sw = sliced_wasserstein(data["prior"],
                                                  data['prior_diffusion'],
                                                  n_slices=1_000)
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    fig.subplots_adjust(left=0, right=1,
                                        bottom=0, top=1)
                    #ax.scatter(*prior_to_plot.T, edgecolors="black", alpha=.2, rasterized=True, c=color_prior)
                    ax.scatter(*data["prior"][:, :2].T, edgecolors="black", alpha=.2, c=color_prior, rasterized=True)
                    ax.scatter(*data["prior_diffusion"][:, :2].T, alpha=.2, edgecolors="black", c=color_diffusion, rasterized=True)
                    ax.set_xlim(-20, 20)
                    ax.set_ylim(-20, 20)

                    fig.savefig(f'images/{component}/{D_X}_{D_Y}_{operator_dim}_{N_components}_{seed}_prior_learning.pdf')
                    #fig.show()
                    plt.close(fig)

                    sw_priors.append({"dist_name": "sw",
                                      "dist_value": prior_sw,
                                      "D_X": D_X,
                                      "D_Y": -1,
                                      "N_components": N_components,
                                      "seed": seed,
                                      "n_ddim_steps": n_steps,
                                      "alg": "diffusion"
                                      })
                    print(make_table(sw_priors))

    for (n_comp, n_steps_ddim), dt in make_table(sw_priors).groupby(['N_components', 'n_ddim_steps']):
        dt['D_X'] = dt['D_X'].apply(int)
        for col in ['diffusion_num']:
            dt[col] = dt[col].astype(float).round(3)
        dt = dt.sort_values(['D_X',], ascending=True)[['D_X', 'N_components', 'diffusion', 'diffusion_num']]
        dt.to_csv(f'data/{component}/{n_comp}_{n_steps_ddim}_diffusion_prior.csv', float_format='%.3f')
