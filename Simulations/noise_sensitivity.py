import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os
from typing import Dict, Tuple, List

from Models.associator import Associator, Trainer
from Models.transforms import rotate

# Define input, hidden, and output sizes
config = dict({
    'input_size': 5,
    # 'hidden_size': 10,
    'hidden_sizes': [5, 12, 100, 1000],
    # 'hidden_sizes': [12],
    'output_size': 5,
    # Extra parameters
    'sample_size': 10,  # data size proxy
    'first_epochs': 20000,  # 1000,
    'slider': False,
    'lr': 0.01,
    'n_reps': 1,
    'test_size': 10,
    # Generalization parameters
    'signal_complexity': 2.0,  # 0 = identity function
    # test_params = [(hidden_size, sample_size)]
    # test_params = [(10**a, sample_size, epochs_*4**(3-a)) for a in range(1, 4)]
})
config['params'] = [(hidden_size, config['sample_size']) for hidden_size in config['hidden_sizes']]
config['results']: Dict[float, Dict] = dict()

max_first_epochs = max([int(min(config['first_epochs'] * 10 ** (3 - np.log10(hidden_size)), 20000))
                        for hidden_size in config['hidden_sizes']])
noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 1 = random noise function

for idx, noise in enumerate(noises):
    snr_results: Dict[Tuple[float, float], List] = dict()
    for hidden_size, sample_size in config['params']:
        firsts = []
        max_epoch = config['first_epochs']
        for n in range(config['n_reps']):
            print(f'Rep: {n+1} - Noise level {idx+1} out of {len(noises)}')
            torch.manual_seed(n)
            # Create an instance of the VectorMapper class
            model = Associator(config['input_size'], hidden_size, config['output_size'], n_conditions=2)

            # Define a loss function and an optimizer
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=config['lr'])

            # Create a Trainer instance
            trainer = Trainer(model, criterion, optimizer)

            # Random A data
            a_train = torch.randn(sample_size, config['input_size'])
            a_test = torch.randn(sample_size, config['input_size'])

            # Rotated B data
            b_data = (1.0 - noise) * rotate(a_train, n_dims=int(config['signal_complexity'])) + \
                     noise * torch.randn(sample_size, config['input_size'])

            # Rotated C data
            c_data = (1.0 - noise) * rotate(a_test, n_dims=int(config['signal_complexity'])) + \
                     noise * torch.randn(sample_size, config['input_size'])

            if config['slider']:
                first_epochs = int(min(config['first_epochs'] * 10 ** (3 - np.log10(hidden_size)), 20000))
            else:
                first_epochs = config['first_epochs']

            # Train the model on associating a_train with B cond=0 while testing a_test on C cond=0
            # This test OOD generalization
            first = trainer.train((a_train, b_data, 0), tests=[(a_test, c_data, 0)], num_epochs=first_epochs)

            epochs_completed, epochs_planned = first['n_epochs']
            for arg in first.keys():
                if 'train' in arg:
                    first[arg] += [first[arg][-1] for _ in range(max_first_epochs - first_epochs)]
                elif 'tests' in arg:
                    for t in range(len(first[arg])):
                        first[arg][t] += [first[arg][t][-1] for _ in range(max_first_epochs - first_epochs)]

            firsts.append(first)
        snr_results[(hidden_size, sample_size)] = firsts
    config['results'][noise] = snr_results

filename = 'NoiseSensitivity_' + str(datetime.now()) + '.pkl'
filepath = os.path.join('../Logs', filename)
with open(filepath, 'wb') as f:
    pickle.dump(config, f)
proxy_artists = []
top_size = np.log10(max(config['hidden_sizes']))
top_noise = max(abs(np.log(np.array(noises)/np.min(noises))))
for jdx, noise in enumerate(config['results'].keys()):
    snr_results = config['results'][noise]
    for (hidden_size, sample_size) in snr_results.keys():
        firsts = snr_results[(hidden_size, sample_size)]
        size_coef = np.log10(hidden_size) / top_size
        base = 0.2
        noise_coef = base + (1-base) * (abs(np.log10(noise)) / top_noise)
        colors = [(size_coef, 0, 0, noise_coef) for _ in range(config['n_reps'])]
        plt.scatter(np.ones(config['n_reps'])*noise*100,
                    1/np.array([first['tests_continuous'][0][-1] for first in firsts]),
                    color=colors)
        if jdx == 0:
            proxy_artists.append(plt.Line2D([0], [0], linestyle='-', color=colors[-1], label=f'{hidden_size} nodes'))
plt.legend(handles=proxy_artists)
plt.xlabel('Noise (%)')
plt.ylabel('Generalization Score (1/loss)')
plt.show()

proxy_artists = []
for jdx, noise in enumerate(config['results'].keys()):
    snr_results = config['results'][noise]
    hidden_size = "NaN"
    for (hidden_size, sample_size) in snr_results.keys():
        firsts = snr_results[(hidden_size, sample_size)]
        size_coef = np.log10(hidden_size) / top_size
        base = 0.2
        noise_coef = base + (1-base) * (abs(np.log10(noise)) / top_noise)
        colors = [(size_coef, 0, 0, noise_coef) for _ in range(config['n_reps'])]
        plt.scatter(np.ones(config['n_reps']) * noise * 100,
                    np.array([first['tests_accuracy'][0][-1] for first in firsts])*100, color=colors)
        # plt.boxplot(noise*100, np.array([first['tests_accuracy'][0][-1] for first in firsts])*100,
        #             color=colors, label=f'{hidden_size} nodes')
        if jdx == 0:
            proxy_artists.append(plt.Line2D([0], [0], linestyle='-', color=colors[-1], label=f'{hidden_size} nodes'))
plt.legend(handles=proxy_artists)
plt.xlabel('Noise (%)')
plt.ylim((0, 100))
plt.ylabel('Generalization Accuracy (%)')
plt.show()
