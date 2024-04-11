import torch
from torch import optim, nn
from datetime import datetime
import pickle
import os

from utils import get_most_recent, make_data
from Visualize.plot_curves import plot_capacity_curves
from Visualize.plot_performance import plot_performance
from Statistics.utils import get_stats
from Models.associator import Associator, Trainer
from Models.transforms import rotate

from_file = True
allow_full_convergence = False

if not from_file:

    # Define input, hidden, and output sizes
    config = dict({
        'input_size': 5,
        # 'hidden_size': 10,
        # 'hidden_sizes': [5, 10, 100, 1000],
        'output_size': 5,
        # Extra parameters
        'sample_size': 10,  # data size proxy
        'context_size': 5,
        'fit_samples': False,
        'fit_models': True,
        'first_epochs': -1,
        'stops': ('delta_train',),
        'delta_min': 10e-5,
        'epoch_min': 5000,
        'model_layers': 2,  # This currently does nothing
        'second_epochs': 10000,  # 2000,
        'lr': 0.01,
        'n_reps': 100,
        # Generalization parameters
        'noise': 0.25,  # then 0.1 then 0.9  # random noise function
        'noisy_generalization': False,
        'signal_complexity': 2.0,  # 0 = identity function
        'criterion': nn.MSELoss,
        'optimizer': optim.SGD
        # test_params = [(hidden_size, sample_size)]
        # test_params = [(10**a, sample_size, epochs_*4**(3-a)) for a in range(1, 4)]
    })
    sample_sizes = None
    sufficient_capacities = None
    if config['fit_samples']:
        config_log = get_most_recent(prefix='Capacity_Fitter', config=config)
        sample_sizes = config_log['data_seed_capacities']
        config['sample_size'] = None
        config['sample_sizes'] = sample_sizes
        config['seeds'] = list(range(config['n_reps']))
    elif config['fit_models']:
        config_log = get_most_recent(prefix='Capacity_Estimator', config=config)
        sufficient_capacities = config_log['model_seed_capacities']
        config['relative_sizes'] = [0.5, 1.0, 10, 100]
        config['hidden_sizes'] = [None for _ in config['relative_sizes']]
        config['sufficient_capacities'] = sufficient_capacities
        config['seeds'] = config_log['seeds']
        if allow_full_convergence:
            config['first_epochs'] = config_log['max_epochs_needed']
        print(f'Max epochs needed: {config_log["max_epochs_needed"]}')
    config['params'] = [(hidden_size, config['sample_size']) for hidden_size in config['hidden_sizes']]
    config['results'] = dict()

    for idx, (hidden_size, sample_size) in enumerate(config['params']):
        firsts = []
        seconds = []
        for i, n in enumerate(config['seeds']):
            print(f'Rep: {i}')
            torch.manual_seed(n)
            if sample_sizes is not None:
                sample_size = sample_sizes[n]
            elif sufficient_capacities is not None:
                hidden_size = int(config['relative_sizes'][idx] * sufficient_capacities[n])
            # Create an instance of the VectorMapper class
            model = Associator(config['input_size'], hidden_size, config['output_size'], n_conditions=2,
                               context_size=config['context_size'])

            # Define a loss function and an optimizer
            criterion = config['criterion']()
            optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

            # Create a Trainer instance
            trainer = Trainer(model, criterion, optimizer)

            # Random A data
            a_train = make_data(sample_size, config['input_size'], n)
            a_test = make_data(sample_size, config['input_size'], n + 2)

            # Rotated B data
            b_data = (1.0 - config['noise']) * rotate(a_train, n_dims=int(config['signal_complexity'])) + \
                     config['noise'] * make_data(sample_size, config['input_size'], n + 1)

            # Rotated C data
            c_data = (1.0 - config['noise']) * rotate(a_test, n_dims=int(config['signal_complexity'])) + \
                     config['noise'] * make_data(sample_size, config['input_size'], n + 3)

            # Rotated C with/without added noise
            if config['noisy_generalization']:
                d_data = c_data
            else:
                d_data = (1.0 - config['noise']) * rotate(a_test, n_dims=int(config['signal_complexity']))

            # Train the model on associating a_train with B cond=0 while testing a_test on C cond=0
            # This test OOD generalization
            first = trainer.train((a_train, b_data, 0), tests=[(a_test, d_data, 0)], num_epochs=config['first_epochs'],
                                  stops=config['stops'], thin=10)

            # Train the model on associating a_train with C cond=1 while testing a_train on B & a_test on C cond=0
            # This tests retention of arbitrary and structured knowledge
            second = trainer.train((a_train, c_data, 1), tests=[(a_train, b_data, 0), (a_test, d_data, 0)],
                                   num_epochs=config['second_epochs'], thin=10)

            epochs_completed, epochs_planned = first['n_epochs']

            firsts.append(first)
            seconds.append(second)
        config['results'][(hidden_size, sample_size)] = firsts, seconds

    filename = 'Generalization_' + str(datetime.now()) + '.pkl'
    filepath = os.path.join('../Logs', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(config, f)
else:
    config = get_most_recent(prefix='Generalization', config={'noise': 0.25, 'noisy_generalization': False})

# ---- Present Results ----

capacities = config['relative_sizes']

# First Plots
# plot_capacity_curves(config, [(0, 'train_accuracy', 0), (0, 'tests_accuracy', 0)], capacities,
#                      'Generalization and Memorization')
# plot_capacity_curves(config, [(0, 'train_continuous', 0), (0, 'tests_continuous', 0)], capacities,
#                      'Generalization and Memorization')

# Second Plots
plot_capacity_curves(config, [(0, 'train_accuracy', 0), (1, 'tests_accuracy', 0)], capacities,
                     'Memorization: Learning and Forgetting', joint=True)
plot_capacity_curves(config, [(0, 'train_continuous', 0), (1, 'tests_continuous', 0)], capacities,
                     'Memorization: Learning and Forgetting', joint=True)
# plot_capacity_curves(config, [(0, 'train_accuracy', 0), (1, 'tests_accuracy', 0)], capacities,
#                      'Memorization: Learning and Forgetting')
# plot_capacity_curves(config, [(0, 'train_continuous', 0), (1, 'tests_continuous', 0)], capacities,
#                      'Memorization: Learning and Forgetting')

# Third Plots
plot_capacity_curves(config, [(0, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
                     'Generalization: Learning and Forgetting', joint=True)
plot_capacity_curves(config, [(0, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
                     'Generalization: Learning and Forgetting', joint=True)
# plot_capacity_curves(config, [(0, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                      'Generalization: Learning and Forgetting')
# plot_capacity_curves(config, [(0, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                      'Generalization: Learning and Forgetting')

# # Fourth Plots
# plot_capacity_curves(config, [(1, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                      'Memorization Over Generalization Forgetting', delta=True)
# plot_capacity_curves(config, [(1, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                      'Memorization Over Generalization Forgetting', delta=True)
#
# # Fifth Plots
# plot_capacity_curves(config, [(1, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                      'Relative Memorization Over Generalization Forgetting', delta=True, ratio=True)
# plot_capacity_curves(config, [(1, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                      'Relative Memorization Over Generalization Forgetting', delta=True, ratio=True)


configs = [
    get_most_recent(prefix='Generalization', config={
        'noisy_generalization': False,
        'noise': noise}) for noise in [0.25, 0.5, 0.75]
]

get_stats(configs, (0, 'train_accuracy', 0), capacities, ['t-statistic', 'p-value', 'cohen-d'])
get_stats(configs, (0, 'tests_accuracy', 0), capacities, ['t-statistic', 'p-value', 'cohen-d'])
get_stats(configs, (1, 'tests_accuracy', 0), capacities, ['t-statistic', 'p-value', 'cohen-d'])
get_stats(configs, (1, 'tests_accuracy', 1), capacities, ['t-statistic', 'p-value', 'cohen-d'])

#
# # First Plots
# plot_performance(configs, [(0, 'train_accuracy', 0), (0, 'tests_accuracy', 0)], capacities,
#                  'Generalization and Memorization', ['Memorization', 'Generalization'])
# plot_performance(configs, [(0, 'train_continuous', 0), (0, 'tests_continuous', 0)], capacities,
#                  'Generalization and Memorization', ['Memorization', 'Generalization'])
#
# Second Plots
plot_performance(configs, [(0, 'train_accuracy', 0), (1, 'tests_accuracy', 0)], capacities,
                 'Memorization: Learning and Forgetting', ['Memorization Learning', 'Memorization Forgetting'])
plot_performance(configs, [(0, 'train_continuous', 0), (1, 'tests_continuous', 0)], capacities,
                 'Memorization: Learning and Forgetting', ['Memorization Learning', 'Memorization Forgetting'])
#
# Third Plots
plot_performance(configs, [(0, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
                 'Generalization: Learning and Forgetting', ['Generalization Learning', 'Generalization Forgetting'])
plot_performance(configs, [(0, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
                 'Generalization: Learning and Forgetting', ['Generalization Learning', 'Generalization Forgetting'])
#
# # Fourth Plots
# plot_performance(configs, [(1, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                  'Memorization Over Generalization Forgetting', [''],
#                  delta=True)
# plot_performance(configs, [(1, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                  'Memorization Over Generalization Forgetting', [''],
#                  delta=True)
#
# # Fifth Plots
# plot_performance(configs, [(1, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                  'Relative Memorization Over Generalization Forgetting', [''],
#                  ratio=True, delta=True)
# plot_performance(configs, [(1, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                  'Relative Memorization Over Generalization Forgetting', [''],
#                  ratio=True, delta=True)
if __name__ == "__main__":
    ...
