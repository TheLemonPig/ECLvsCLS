import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def plot_capacity_curves(config, curves: List[Tuple[int, str, int]], capacities: List[float], title: str, delta=False):
    n_epochs = curves[0][0] * config['second_epochs'] + (1 - curves[0][0]) * config['first_epochs']
    n_capacities = len(capacities)
    n_curves = len(curves)
    curves_array = np.zeros((n_curves, n_capacities, n_epochs))
    for i, parameters in enumerate(config['results']):
        # TODO: Fix parameters to actually be the relative capacities
        results = config['results'][parameters]
        for n in range(config['n_reps']):
            rep_results = results[0][n], results[1][n]
            for j, curve in enumerate(curves):
                curve_run, curve_stat, curve_num = curve
                if curve_stat.startswith('train'):
                    curves_array[j, i, :] += np.array(rep_results[curve_run][curve_stat]) / config['n_reps']
                else:
                    curves_array[j, i, :] += np.array(rep_results[curve_run][curve_stat][curve_num]) / config['n_reps']
    curve_stat = curves[0][1]
    if curve_stat.endswith('accuracy'):
        curves_array = curves_array * 100
    else:
        curves_array = -np.log(curves_array)
    if delta:
        assert n_curves == 2, 'cannot create delta for more than 2 curves in a single plot'
        curves_array = (curves_array[0] - curves_array[1]).reshape((1, n_capacities, n_epochs))
        n_curves = 1
    for j in range(n_curves):
        color = [0, 0, 0, 1]
        for i, capacity in enumerate(capacities):
            color[j] = i / (len(capacities) - 1)
            capacity_curve = curves_array[j, i]
            plt.plot(np.arange(len(capacity_curve)), capacity_curve, label=f'{capacities[i]} x Sufficient Capacity',
                     c=tuple(color))
    if curve_stat.endswith('accuracy'):
        plt.ylabel('Accuracy (%)')
    else:
        plt.ylabel('Performance (-log[continuous loss])')
    plt.xlabel("Epochs")
    plt.legend()
    plt.title(title)
    plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
    plt.show()
