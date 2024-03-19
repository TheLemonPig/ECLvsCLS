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
    'output_size': 5,
    'sample_size': 10,
    'model_layers': 2,
    # Extra parameters
    'first_epochs': 20000,
    'second_epochs': 2000,
    'lr': 0.01,
    'n_reps': 20,
    'test_size': 10,
    # Generalization parameters
    'noise': 0.5,  # 1 = random noise function
    'signal_complexity': 2.0,  # 0 = identity function
    'criterion': nn.MSELoss,
    'optimizer': optim.SGD
})
config['seeds'] = list(range(config['n_reps']))
start_size = 5
data_capacities = dict({k: -1 for k in range(config['n_reps'])})
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
        model = Associator(config['input_size'], hidden_size, config['output_size'])

        # Define a loss function and an optimizer
        criterion = config['criterion']()
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

        # Create a Trainer instance
        trainer = Trainer(model, criterion, optimizer)

        # Random x data
        x = torch.randn(config['sample_size'], config['input_size'])

        # Rotated and noise-embedded y data
        y = (1.0 - config['noise']) * rotate(x, n_dims=int(config['signal_complexity'])) + \
            config['noise'] * torch.randn(config['sample_size'], config['input_size'])

        results = trainer.train((x, y, 0), num_epochs=config['first_epochs'])
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
    data_capacities[seed] = sufficient_capacity

config['seed_capacities'] = data_capacities
print(data_capacities)

filename = 'Capacity_Estimator_' + str(datetime.now()) + '.pkl'
filepath = os.path.join('../Logs', filename)
with open(filepath, 'wb') as f:
    pickle.dump(config, f)

if __name__ == "__main__":
    ...
