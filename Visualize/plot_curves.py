import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def plot_capacity_curves(config, curves: List[Tuple[int, str, int]], capacities: List[float], title: str, delta=False,
                         ratio=False, y_lim: Tuple[float, float] = None, joint=False):
    if joint:
        n_epochs = config['first_epochs'] + config['second_epochs']
        assert len(curves) == 2, 'cannot join more than 2 curves in a single plot'
        n_curves = 1
    else:
        n_epochs = curves[0][0] * config['second_epochs'] + (1 - curves[0][0]) * config['first_epochs']
        n_curves = len(curves)
    n_capacities = len(capacities)
    min_epochs = n_epochs
    curves_array = np.zeros((n_curves, n_capacities, n_epochs))
    for i, parameters in enumerate(config['results']):
        # TODO: Fix parameters to actually be the relative capacities
        results = config['results'][parameters]
        for n in range(config['n_reps']):
            rep_results = results[0][n], results[1][n]
            if joint:
                joint_curve_list = []
                for curve in curves:
                    curve_run, curve_stat, curve_num = curve
                    if curve_stat.startswith('train'):
                        joint_curve_list += rep_results[curve_run][curve_stat]
                    else:
                        joint_curve_list += rep_results[curve_run][curve_stat][curve_num]
                curve_vector = np.array(joint_curve_list)
                curves_array[0, i] += curve_vector / config['n_reps']
            else:
                for j, curve in enumerate(curves):
                    curve_run, curve_stat, curve_num = curve
                    if curve_stat.startswith('train'):
                        curve_vector = np.array(rep_results[curve_run][curve_stat])
                    else:
                        curve_vector = np.array(rep_results[curve_run][curve_stat][curve_num])
                    len_vector = curve_vector.shape[0]
                    min_epochs = min(min_epochs, len_vector)
                    curves_array[j, i, :len_vector] += curve_vector / config['n_reps']
    curve_stat = curves[0][1]
    curves_array = curves_array[:, :, :min_epochs]
    if ratio:
        curves_array = curves_array / np.tile(curves_array[:, :, 0][:, :, np.newaxis], [1, 1, curves_array.shape[2]])
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
    if ratio:
        plt.ylabel('Relative')
    elif curve_stat.endswith('accuracy'):
        plt.ylabel('Accuracy (%)')
    else:
        plt.ylabel('Performance (-log[MSE])')
    if y_lim is None:
        if curve_stat.endswith('accuracy'):
            y_lim = (0, 100)
        else:
            y_lim = (-2, 10)
    if joint:
        plt.axvline(x=config['first_epochs'], ymin=y_lim[0], ymax=y_lim[1],
                    linestyle='--', color='gray', label='Second Trial')
    plt.ylim(y_lim)
    plt.xlabel("Epochs")
    plt.legend()
    plt.title(title)
    plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
    plt.show()
