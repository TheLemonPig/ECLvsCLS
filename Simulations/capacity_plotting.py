import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from Models.associator import Associator, Trainer

# Define input, hidden, and output sizes
#input_sizes = [1, 2, 3, 4, 5]
input_sizes = [4]
hidden_sizes = list(range(10, 20))  # model capacity proxy
#output_sizes = [1, 2, 3, 4, 5]
output_sizes = input_sizes
context_size = 2

# Extra parameters
n_reps = 3
p = 0.05
epochs_ = -1
lr_ = 0.01

dim_dic = {'param_losses'}
for i in range(len(input_sizes)):
    input_size = input_sizes[i]
    output_size = output_sizes[i]
    capacities = dict()  # This should get overwritten
    capacity = dict()
    for hidden_size in hidden_sizes:
        capacity_reached = False
        sample_size = 1
        for sample_size in range(int(hidden_size * 1.3), 3*hidden_size, 1):
            print(f'\n---Training {hidden_size}-node model on {sample_size} {input_size}-D samples---')
            total_losses = []
            strike = False
            broke = False
            n = 0
            for n in range(n_reps):
                if broke:
                    break
                print(f'Repetition: {n+1}/{n_reps} Strike: {strike}')
                torch.manual_seed(n)
                # Create an instance of the VectorMapper class
                model = Associator(input_size, hidden_size, output_size, context_size)

                # Define a loss function and an optimizer
                criterion = nn.MSELoss()
                optimizer = optim.SGD(model.parameters(), lr=lr_)

                # Create a Trainer instance
                trainer = Trainer(model, criterion, optimizer)

                # Random A data
                a_data = torch.randn(sample_size, input_size)

                # Random B data
                b_data = torch.randn(sample_size, output_size)

                # Set B condition
                a_data[:, -context_size:] = 0
                # Train the model on associating A with B
                trainer.train(a_data, b_data, num_epochs=epochs_, print_every=1000)
                total_losses.append(trainer.losses)
                # Abort if more than one instance of memorization was unsuccessful
                if trainer.losses[-1] < 1.0:
                    if strike:
                        broke = True
                        break
                    else:
                        strike = True
            if broke:
                capacity_reached = True
                break
        print(f'{hidden_size}-node model exceeded capacity on {sample_size} samples')
        capacity[hidden_size] = sample_size - int(capacity_reached)
    capacities[(input_size, output_size)] = capacity

print(capacities)
for params in capacities.keys():
    param_capacities = capacities[params]
    capacity_array = np.array([[k, v] for k, v in param_capacities.items()])
    plt.plot(capacity_array[:, 0], capacity_array[:, 1], label=f"Input/Output size: {params}")
plt.show()

if __name__ == "__main__":
    ...
