import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

from Models.associator import Associator, Trainer
from Models.transforms import rotate

from utils import make_data

# Define input, hidden, and output sizes
config = dict({
    'input_size': 5,
    'output_size': 5,
    'sample_size': 10,
    'context_size': 5,
    'model_layers': 2,
    # Extra parameters
    'first_epochs': -1,
    'stops': ('delta_train', 'train_accuracy'),
    'delta_min': 10e-5,
    'epoch_min': 5000,
    'second_epochs': 2000,
    'lr': 0.001,
    'n_reps': 20,
    # Generalization parameters
    'noise': 0.1,  # 1 = random noise function
    'signal_complexity': 2.0,  # 0 = identity function
    'criterion': nn.MSELoss,
    'optimizer': optim.SGD
})
config['seeds'] = list(range(config['n_reps']))
start_size = 10
seed_capacities = dict({k: -1 for k in range(config['n_reps'])})
for seed in config['seeds']:
    print(f'Finding Sufficient Capacity for seed={seed}')
    seed_capacity = dict()
    hidden_size = start_size
    while (True not in seed_capacity.values()) or (False not in seed_capacity.values()):
        if len(seed_capacity) > 0 and seed_capacity[hidden_size] is True:
            print(f'Sufficient Capacity <= {hidden_size}')
            hidden_size -= 1
        elif len(seed_capacity) > 0 and seed_capacity[hidden_size] is False:
            print(f'Sufficient Capacity > {hidden_size}')
            hidden_size += 1

        torch.manual_seed(seed)
        # Create an instance of the VectorMapper class
        model = Associator(config['input_size'], hidden_size, config['output_size'], context_size=config['context_size'])

        # Define a loss function and an optimizer
        criterion = config['criterion']()
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

        # Create a Trainer instance
        trainer = Trainer(model, criterion, optimizer)

        # Random x data
        x = make_data(config['sample_size'], config['input_size'], seed)

        # Rotated and noise-embedded y data
        y = (1.0 - config['noise']) * rotate(x, n_dims=int(config['signal_complexity'])) + \
            config['noise'] * make_data(config['sample_size'], config['input_size'], seed+1)

        results = trainer.train((x, y, 0), num_epochs=config['first_epochs'], stops=config['stops'],
                                epoch_min=config['epoch_min'])
        seed_capacity[hidden_size] = (results['train_accuracy'][-1] == 1.0)
    sufficient_capacity = start_size
    for hidden_size in seed_capacity.keys():
        if seed_capacity[hidden_size]:
            if seed_capacity[sufficient_capacity]:
                if hidden_size < sufficient_capacity:
                    sufficient_capacity = hidden_size
            else:
                sufficient_capacity = hidden_size
    print(f'Sufficient Capacity = {sufficient_capacity}\n')
    seed_capacities[seed] = sufficient_capacity

config['model_seed_capacities'] = seed_capacities
print(seed_capacities)

filename = 'Capacity_Estimator_' + str(datetime.now()) + '.pkl'
filepath = os.path.join('../Logs', filename)
with open(filepath, 'wb') as f:
    pickle.dump(config, f)

if __name__ == "__main__":
    ...
