import torch
from torch import optim, nn
from datetime import datetime
import pickle
import os
import copy

from utils import get_most_recent, make_data
from Visualize.plot_curves import plot_capacity_curves
from Visualize.plot_performance import plot_performance
from Models.associator import Associator, Trainer
from Models.transforms import rotate

from_file = True

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
        'first_epochs': 200000,
        'stops': ('epochs', 'delta_train'),
        'delta_min': 10e-5,
        'epoch_min': 5000,
        'model_layers': 2,  # This currently does nothing
        'second_epochs': 2000,  #2000,
        'lr': 0.001,
        'n_reps': 20,
        # Generalization parameters
        'noise': 0.1,  # 1 = random noise function
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
            a_test = make_data(sample_size, config['input_size'], n+2)

            # Rotated B data
            b_data = (1.0 - config['noise']) * rotate(a_train, n_dims=int(config['signal_complexity'])) + \
                     config['noise'] * make_data(sample_size, config['input_size'], n+1)

            # Rotated C data
            c_data = (1.0 - config['noise']) * rotate(a_test, n_dims=int(config['signal_complexity'])) + \
                     config['noise'] * make_data(sample_size, config['input_size'], n+3)

            # Train the model on associating a_train with B cond=0 while testing a_test on C cond=0
            # This test OOD generalization
            first = trainer.train((a_train, b_data, 0), tests=[(a_test, c_data, 0)], num_epochs=config['first_epochs'],
                                  stops=config['stops'])

            # Train the model on associating a_train with C cond=1 while testing a_train on B & a_test on C cond=0
            # This tests retention of arbitrary and structured knowledge
            second = trainer.train((a_train, c_data, 1), tests=[(a_train, b_data, 0), (a_test, c_data, 0)],
                                   num_epochs=config['second_epochs'])

            epochs_completed, epochs_planned = first['n_epochs']

            firsts.append(first)
            seconds.append(second)
        config['results'][(hidden_size, sample_size)] = firsts, seconds

    filename = 'Generalization_' + str(datetime.now()) + '.pkl'
    filepath = os.path.join('../Logs', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(config, f)
else:
    config = get_most_recent(prefix='Generalization')

# ---- Present Results ----

capacities = config['relative_sizes']

# First Plots
plot_capacity_curves(config, [(0, 'train_accuracy', 0), (0, 'tests_accuracy', 0)], capacities,
                     'Generalization and Memorization')
plot_capacity_curves(config, [(0, 'train_continuous', 0), (0, 'tests_continuous', 0)], capacities,
                     'Generalization and Memorization')

plot_performance(config, [(0, 'train_accuracy', 0), (0, 'tests_accuracy', 0)], capacities,
                 'Generalization and Memorization', ['Memorization', 'Generalization'])
plot_performance(config, [(0, 'train_continuous', 0), (0, 'tests_continuous', 0)], capacities,
                 'Generalization and Memorization', ['Memorization', 'Generalization'])

temp_config = copy.deepcopy(config)
temp_config['noise'] = 0.5
other_config = get_most_recent(prefix='Generalization', config=temp_config)

configs = [config, other_config]

plot_performance(configs, [(0, 'train_accuracy', 0), (0, 'tests_accuracy', 0)], capacities,
                 'Generalization and Memorization', ['Memorization', 'Generalization'])
plot_performance(configs, [(0, 'train_continuous', 0), (0, 'tests_continuous', 0)], capacities,
                 'Generalization and Memorization', ['Memorization', 'Generalization'])
# # Second Plots
# plot_capacity_curves(config, [(1, 'train_accuracy', 0), (1, 'tests_accuracy', 0)], capacities,
#                      'Memorization Forgetting')
# plot_capacity_curves(config, [(1, 'train_continuous', 0), (1, 'tests_continuous', 0)], capacities,
#                      'Memorization Forgetting')
#
# # Third Plots
# plot_capacity_curves(config, [(1, 'train_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                      'Generalization Forgetting')
# plot_capacity_curves(config, [(1, 'train_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                      'Generalization Forgetting')
#
# # Fourth Plots
# plot_capacity_curves(config, [(1, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                      'Memorization VS Generalization Forgetting')
# plot_capacity_curves(config, [(1, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                      'Memorization VS Generalization Forgetting')
#
# # Fifth Plots
# plot_capacity_curves(config, [(1, 'tests_accuracy', 0), (1, 'tests_accuracy', 1)], capacities,
#                      'Memorization Over Generalization Forgetting', delta=True)
# plot_capacity_curves(config, [(1, 'tests_continuous', 0), (1, 'tests_continuous', 1)], capacities,
#                      'Memorization Over Generalization Forgetting', delta=True)


if __name__ == "__main__":
    ...
