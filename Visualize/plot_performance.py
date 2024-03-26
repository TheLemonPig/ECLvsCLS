import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_performance(configs, curves: List[Tuple[int, str, int]], capacities: List[float], title: str, labels: List[str],
                     delta=False):
    n_reps = configs[0]['n_reps']
    n_capacities = len(capacities)
    n_curves = len(curves)
    curves_array = np.zeros((n_curves, n_capacities, n_reps))
    for config in configs:
        for i, parameters in enumerate(config['results']):
            # TODO: Fix parameters to actually be the relative capacities
            results = config['results'][parameters]
            for n in range(config['n_reps']):
                rep_results = results[0][n], results[1][n]
                for j, curve in enumerate(curves):
                    curve_run, curve_stat, curve_num = curve
                    if curve_stat.startswith('train'):
                        curves_array[j, i, n] += rep_results[curve_run][curve_stat][-1]
                    else:
                        curves_array[j, i, n] += rep_results[curve_run][curve_stat][curve_num][-1]
        curve_stat = curves[0][1]
        if curve_stat.endswith('accuracy'):
            curves_array = curves_array * 100
        else:
            curves_array = -np.log(curves_array)
        if delta:
            assert n_curves == 2, 'cannot create delta for more than 2 curves in a single plot'
            curves_array = (curves_array[0] - curves_array[1]).reshape((1, n_capacities, n_reps))
            n_curves = 1
        for j in range(n_curves):
            color = [0, 0, 0, 1]
            color[j] = config['noise'] * 0.5 + 0.5
            mean_performances = curves_array[j].mean(axis=1)
            plt.plot(np.log10(capacities), mean_performances, label=f"{labels[j]} Noise={100*config['noise']}%",
                     c=tuple(color))
            standard_error = np.std(curves_array[j], axis=1) / np.sqrt(config['n_reps'])
            plt.errorbar(np.log10(capacities), mean_performances, yerr=standard_error, fmt='-o', capsize=5,
                         label='Standard Error', c=tuple(color))
        if curve_stat.endswith('accuracy'):
            plt.ylabel('Accuracy (%)')
            plt.axvline(x=0, ymin=0, ymax=100, linestyle='--', color='gray', label='Sufficient Capacity')
        else:
            plt.ylabel('Performance (-log[continuous loss])')
            plt.axvline(x=0, ymin=0, ymax=10, linestyle='--', color='gray', label='Sufficient Capacity')
    plt.xlabel("Capacity (10^n x sufficient capacity)")
    plt.legend()
    plt.title(title)
    plt.show()
