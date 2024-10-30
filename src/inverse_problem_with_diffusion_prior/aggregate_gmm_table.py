import pandas as pd
import numpy as np

data = pd.read_csv('data/gmm_inverse_problem_comparison.csv')
agg_data = data.groupby(['dim', 'dim_y', 'n_steps', 'algorithm', 'distance_name']).distance.apply(lambda x: f'{np.nanmean(x):.1f} ± {1.96 * np.nanstd(x) / (x.shape[0]**.5):.1f}').reset_index()

agg_data_sw = agg_data.loc[agg_data.distance_name == 'sw'].pivot(index=('dim', 'dim_y', 'n_steps'), columns='algorithm', values=['distance']).reset_index()
agg_data_sw.columns = [col[-1].replace(' ', '_') if col[-1] else col[0].replace(' ', '_') for col in agg_data_sw.columns.values]

for col in agg_data_sw.columns:
    if col not in ['dim', 'dim_y', 'n_steps']:
        agg_data_sw[col + '_num'] = agg_data_sw[col].apply(lambda x: float(x.split(" ± ")[0]))
agg_data_sw.loc[agg_data_sw.n_steps == 20].to_csv('data/gmm_inverse_problem_aggregated_sliced_wasserstein_20_steps.csv')
agg_data_sw.loc[agg_data_sw.n_steps == 100].to_csv('data/gmm_inverse_problem_aggregated_sliced_wasserstein_100_steps.csv')
