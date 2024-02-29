import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

from Models.associator import Associator, Trainer
from Models.transforms import rotate

# Define input, hidden, and output sizes
config = dict({
    'input_size': 5,
    'hidden_size': 10,
    'output_size': 5,
    # Extra parameters
    'sample_size': 10,  # data size proxy
    'first_epochs': 200,
    'second_epochs': 200,
    'lr': 0.01,
    'n_reps': 2,
    'test_size': 10,
    # Generalization parameters
    'signal_noise_ratio': 0.0,
    'signal_complexity': 0.0,  # 0 = identity function
    # test_params = [(hidden_size, sample_size)]
    # test_params = [(10**a, sample_size, epochs_*4**(3-a)) for a in range(1, 4)]
})
config['test_params'] = [(a, config['sample_size']) for a in [5, 12, 100, 1000]]
config['results'] = dict()

for hidden_size, sample_size in config['test_params']:
    firsts = []
    seconds = []
    for n in range(config['n_reps']):
        torch.manual_seed(n)
        # Create an instance of the VectorMapper class
        model = Associator(config['input_size'], config['hidden_size'], config['output_size'], n_conditions=2)

        # Define a loss function and an optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])

        # Create a Trainer instance
        trainer = Trainer(model, criterion, optimizer)

        # Random A data
        a_train = torch.randn(sample_size, config['input_size'])
        a_test = torch.randn(sample_size, config['input_size'])

        # Rotated B data
        b_data = rotate(a_train, n_dims=int(config['signal_complexity']))

        # Rotated C data
        c_data = rotate(a_test, n_dims=int(config['signal_complexity']))

        # Train the model on associating a_train with B cond=0 while testing a_test on C cond=0
        # This test OOD generalization
        # TODO: Match updated classifiers
        first = trainer.train((a_train, b_data, 0), tests=[(a_test, c_data, 0)], num_epochs=config['first_epochs'])

        # Train the model on associating a_train with C cond=1 while testing a_train on B & a_test on C cond=0
        # This tests retention of arbitrary and structured knowledge
        # TODO: Match updated classifier
        second = trainer.train((a_train, c_data, 1), tests=[(a_train, b_data, 0), (a_test, c_data, 0)],
                               num_epochs=config['second_epochs'])
        firsts.append(first)
        seconds.append(second)
    config['results'][(hidden_size, sample_size)] = firsts, seconds

filename = 'Generalization_' + str(datetime.now()) + '.pkl'
filepath = os.path.join('../Logs', filename)
with open(filepath, 'wb') as f:
    pickle.dump(config, f)


# ---- Present Results ----
# epochs = np.arange(epochs_)
# plt.figure(figsize=(10, 5))
# top = np.log10(max(test_params[-1]))
# for idx, run in enumerate(test_params):
#     coef = np.log10(run[0]) / top
#     plt.plot(epochs, c_losses[idx], label=f'C Training {run[1]} samples {run[0]} nodes', c=(0, 0, coef, 1))
#     plt.plot(epochs, b_losses[idx], label=f'B Testing {run[1]} samples {run[0]} nodes', c=(coef, 0, 0, 1))
# plt.axhline(y=1, color='gray', linestyle='--', label='100% Memorization')
# plt.title('Average Classification Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

if __name__ == "__main__":
    ...
