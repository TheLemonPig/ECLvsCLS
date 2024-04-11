import numpy as np
from typing import List, Tuple
import scipy


def get_stats(configs, curve: Tuple[int, str, int], capacities: List[float], stats=None):
    print(f'\n\nCurve: {curve}\n\n')
    for config in configs:
        print(f'\nNoise: {100 * config["noise"]}%\n')
        n_reps = config['n_reps']
        n_capacities = len(capacities)
        curves_array = np.zeros((n_capacities, n_reps))
        for i, parameters in enumerate(config['results']):
            # TODO: Fix parameters to actually be the relative capacities
            results = config['results'][parameters]
            for n in range(config['n_reps']):
                rep_results = results[0][n], results[1][n]
                curve_run, curve_stat, curve_num = curve
                if curve_stat.startswith('train'):
                    curves_array[i, n] += rep_results[curve_run][curve_stat][-1] / config['n_reps']
                else:
                    curves_array[i, n] += rep_results[curve_run][curve_stat][curve_num][-1] / config['n_reps']
        curve_stat = curve[1]
        if curve_stat.endswith('accuracy'):
            curves_array = curves_array * 100
        else:
            curves_array = -np.log(curves_array)
        # print(curves_array)
        for idx in range(n_capacities):
            for jdx in range(idx):
                t_statistic, p_value = scipy.stats.ttest_ind(curves_array[idx], curves_array[jdx])
                coh_d = cohen_d(curves_array[idx], curves_array[jdx])
                stats_str = f'Capacity {capacities[idx]} vs Capacity {capacities[jdx]} -- '
                if 't-statistic' in stats:
                    stats_str += f't-statistic: {t_statistic} '
                if 'p-value' in stats:
                    stats_str += f'p-value: {p_value} '
                if 'cohen-d' in stats:
                    stats_str += f'cohen-d: {coh_d} '
                print(stats_str)


def cohen_d(group1, group2):
    """
    Calculate Cohen's d effect size for two groups.

    Parameters:
        group1 (array-like): Data for group 1.
        group2 (array-like): Data for group 2.

    Returns:
        float: Cohen's d effect size.
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    d = mean_diff / pooled_std
    return d
