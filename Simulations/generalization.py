import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

from utils import get_most_recent, make_data
from Models.associator import Associator, Trainer
from Models.transforms import rotate

# Define input, hidden, and output sizes
config = dict({
    'input_size': 5,
    # 'hidden_size': 10,
    'hidden_sizes': [5, 10, 100, 1000],
    'output_size': 5,
    # Extra parameters
    'sample_size': 10,  # data size proxy
    'fit_samples': False,
    'fit_models': True,
    'first_epochs': -1,
    'stops': ('delta_train',),
    'second_epochs': 2000,  #2000,
    'lr': 0.001,
    'n_reps': 10,
    # Generalization parameters
    'noise': 0.5,  # 1 = random noise function
    'signal_complexity': 2.0,  # 0 = identity function
    'criterion': nn.MSELoss,
    'optimizer': optim.Adam
    # test_params = [(hidden_size, sample_size)]
    # test_params = [(10**a, sample_size, epochs_*4**(3-a)) for a in range(1, 4)]
})
sample_sizes = None
sufficient_capacities = None
if config['fit_samples']:
    # TODO: Get get_most_recent to find most recent where all config parameters match
    config_log = get_most_recent(prefix='Capacity_Fitter', config=config)
    sample_sizes = config_log['data_seed_capacities']
    config['sample_size'] = None
    config['sample_sizes'] = sample_sizes
elif config['fit_models']:
    config_log = get_most_recent(prefix='Capacity_Estimator', config=config)
    sufficient_capacities = config_log['model_seed_capacities']
    config['hidden_sizes'] = [None for _ in config['hidden_sizes']]
    config['sufficient_capacities'] = sufficient_capacities
    config['relative_sizes'] = [0.5, 1.0, 10, 100]

config['params'] = [(hidden_size, config['sample_size']) for hidden_size in config['hidden_sizes']]
config['results'] = dict()

for idx, (hidden_size, sample_size) in enumerate(config['params']):
    firsts = []
    seconds = []
    for n in range(config['n_reps']):
        print(f'Rep: {n}')
        torch.manual_seed(n)
        if sample_sizes is not None:
            sample_size = sample_sizes[n]
        elif sufficient_capacities is not None:
            hidden_size = int(config['relative_sizes'][idx] * sufficient_capacities[n])
            assert hidden_size > 2, TypeError('No space for condition labels')
        # Create an instance of the VectorMapper class
        model = Associator(config['input_size'], hidden_size, config['output_size'], n_conditions=2)

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


# ---- Present Results ----

top = np.log10(max(config['hidden_sizes']))

# --- Plot 1 ---

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_b_0_avg_0 = np.zeros((config['first_epochs'],))
    a_test_c_0_avg_0 = np.zeros((config['first_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]
        a_train_b_0_avg_0 += np.array(first['train_accuracy']) / config['n_reps']
        a_test_c_0_avg_0 += np.array(first['tests_accuracy'][0]) / config['n_reps']
    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_b_0_avg_0)), a_train_b_0_avg_0*100, label=f'b=0 {params[0]} nodes',
                  c=(0, 0, coef, 1))
    plt.plot(np.arange(len(a_test_c_0_avg_0)), a_test_c_0_avg_0*100, label=f'c=0 {params[0]} nodes',
                  c=(coef, 0, 0, 1))
plt.ylabel('Accuracy (%)')
plt.xlabel("Epochs")
plt.legend()
plt.title('first: generalization learning')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_b_0_avg_0 = np.zeros((config['first_epochs'],))
    a_test_c_0_avg_0 = np.zeros((config['first_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]
        a_train_b_0_avg_0 += np.array(first['train_continuous']) / config['n_reps']
        a_test_c_0_avg_0 += np.array(first['tests_continuous'][0]) / config['n_reps']

    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_b_0_avg_0)), -np.log(a_train_b_0_avg_0), label=f'b=0 {params[0]} nodes',
                  c=(0, 0, coef, 1))
    plt.plot(np.arange(len(a_test_c_0_avg_0)), -np.log(a_test_c_0_avg_0), label=f'c=0 {params[0]} nodes',
                  c=(coef, 0, 0, 1))
plt.ylabel('Performance (-log[continuous loss])')
plt.xlabel("Epochs")
plt.legend()
plt.title('first: generalization learning')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

# --- Plot 2 ---
for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_c_1_avg_1 = np.zeros((config['second_epochs'],))
    a_train_b_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]
        a_train_c_1_avg_1 += np.array(second['train_accuracy']) / config['n_reps']
        a_train_b_0_avg_1 += np.array(second['tests_accuracy'][0]) / config['n_reps']
    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_c_1_avg_1)), a_train_c_1_avg_1*100, label=f'c=1 {params[0]} nodes',
                  c=(0, 0, coef, 1))
    plt.plot(np.arange(len(a_train_b_0_avg_1)), a_train_b_0_avg_1*100, label=f'b=0 {params[0]} nodes',
                  c=(coef, 0, 0, 1))

plt.ylabel('Accuracy (%)')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: memorization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_c_1_avg_1 = np.zeros((config['second_epochs'],))
    a_train_b_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]
        a_train_c_1_avg_1 += np.array(second['train_continuous']) / config['n_reps']
        a_train_b_0_avg_1 += np.array(second['tests_continuous'][0]) / config['n_reps']
    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_c_1_avg_1)), -np.log(a_train_c_1_avg_1), label=f'c=1 {params[0]} nodes',
                  c=(0, 0, coef, 1))
    plt.plot(np.arange(len(a_train_b_0_avg_1)), -np.log(a_train_b_0_avg_1), label=f'b=0 {params[0]} nodes',
                  c=(coef, 0, 0, 1))

plt.ylabel('Performance (-log[continuous loss])')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: memorization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

# --- Plot 3 ---

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_c_1_avg_1 = np.zeros((config['second_epochs'],))
    a_test_c_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]

        a_train_c_1_avg_1 += np.array(second['train_accuracy']) / config['n_reps']
        a_test_c_0_avg_1 += np.array(second['tests_accuracy'][1]) / config['n_reps']
    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_c_1_avg_1)), a_train_c_1_avg_1*100, label=f'a_train_c_1_avg_1 {params[0]} nodes',
                  c=(0, 0, coef, 1))
    plt.plot(np.arange(len(a_test_c_0_avg_1)), a_test_c_0_avg_1*100, label=f'a_test_c_0_avg_1 {params[0]} nodes',
                  c=(0, coef, 0, 1))

plt.ylabel('Accuracy (%)')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: generalization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_c_1_avg_1 = np.zeros((config['second_epochs'],))
    a_test_c_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]

        a_train_c_1_avg_1 += np.array(second['train_continuous']) / config['n_reps']
        a_test_c_0_avg_1 += np.array(second['tests_continuous'][1]) / config['n_reps']
    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_c_1_avg_1)), -np.log(a_train_c_1_avg_1), label=f'a_train_c_1_avg_1 {params[0]} nodes',
                  c=(0, 0, coef, 1))
    plt.plot(np.arange(len(a_test_c_0_avg_1)), -np.log(a_test_c_0_avg_1), label=f'a_test_c_0_avg_1 {params[0]} nodes',
                  c=(0, coef, 0, 1))

plt.ylabel('Performance (-log[continuous loss])')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: generalization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

# --- Plot 4 ---

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_b_0_avg_1 = np.zeros((config['second_epochs'],))
    a_test_c_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]

        a_train_b_0_avg_1 += np.array(second['tests_accuracy'][0]) / config['n_reps']
        a_test_c_0_avg_1 += np.array(second['tests_accuracy'][1]) / config['n_reps']
    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_b_0_avg_1)), a_train_b_0_avg_1*100, label=f'a_train_b_0_avg_1 {params[0]} nodes',
                  c=(coef, 0, 0, 1))
    plt.plot(np.arange(len(a_test_c_0_avg_1)), a_test_c_0_avg_1*100, label=f'a_test_c_0_avg_1 {params[0]} nodes',
                  c=(0, coef, 0, 1))

plt.ylabel('Accuracy (%)')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: memorization vs generalization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_b_0_avg_1 = np.zeros((config['second_epochs'],))
    a_test_c_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]

        a_train_b_0_avg_1 += np.array(second['tests_continuous'][0]) / config['n_reps']
        a_test_c_0_avg_1 += np.array(second['tests_continuous'][1]) / config['n_reps']
    coef = np.log10(params[0]) / top
    plt.plot(np.arange(len(a_train_b_0_avg_1)), -np.log(a_train_b_0_avg_1), label=f'a_train_b_0_avg_1 {params[0]} nodes',
                  c=(coef, 0, 0, 1))
    plt.plot(np.arange(len(a_test_c_0_avg_1)), -np.log(a_test_c_0_avg_1), label=f'a_test_c_0_avg_1 {params[0]} nodes',
                  c=(0, coef, 0, 1))

plt.ylabel('Performance (-log[continuous loss])')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: memorization vs generalization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

# --- Plot 5 ---

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_b_0_avg_1 = np.zeros((config['second_epochs'],))
    a_test_c_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]

        a_train_b_0_avg_1 += np.array(second['tests_accuracy'][0]) / config['n_reps']
        a_test_c_0_avg_1 += np.array(second['tests_accuracy'][1]) / config['n_reps']
    coef = np.log10(params[0]) / top
    delta_curve = a_test_c_0_avg_1*100 - a_train_b_0_avg_1*100
    plt.plot(np.arange(len(a_train_b_0_avg_1)), delta_curve,
             label=f'a_train_c_0_avg_1-a_train_b_0_avg_1 {params[0]} nodes',
             c=(coef, 0, 0, 1))

plt.ylabel('Accuracy (%)')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: generalization over memorization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

for params in config['results']:
    firsts, seconds = config['results'][params]

    a_train_b_0_avg_1 = np.zeros((config['second_epochs'],))
    a_test_c_0_avg_1 = np.zeros((config['second_epochs'],))
    for n in range(config['n_reps']):
        first = firsts[n]
        second = seconds[n]

        a_train_b_0_avg_1 += np.array(second['tests_continuous'][0]) / config['n_reps']
        a_test_c_0_avg_1 += np.array(second['tests_continuous'][1]) / config['n_reps']
    coef = np.log10(params[0]) / top
    delta_curve = -np.log(a_test_c_0_avg_1) + np.log(a_train_b_0_avg_1)
    plt.plot(np.arange(len(a_train_b_0_avg_1)), delta_curve,
             label=f'a_train_c_0_avg_1 - a_train_b_0_avg_1 {params[0]} nodes',
             c=(coef, 0, 0, 1))

plt.ylabel('Performance (-log[continuous loss])')
plt.xlabel("Epochs")
plt.legend()
plt.title('second: generalization over memorization forgetting')
plt.suptitle(f"Noise: {config['noise']}, N_reps: {config['n_reps']}")
plt.show()

if __name__ == "__main__":
    ...
