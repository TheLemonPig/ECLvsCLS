import torch
from torch import optim, nn

from Models.associator import Associator, Trainer

# Define input, hidden, and output sizes
input_size = 5
hidden_size = 25  # model capacity proxy
output_size = 5
context_size = 2

# Extra parameters
n_reps = 3
p = 0.05
sample_size = 10  # data size proxy
epochs_ = -1
lr_ = 0.01

# test_params = [(hidden_size, sample_size)]
test_params = [(hidden_size, c) for c in [2*hidden_size, int(2.2*hidden_size), int(2.4*hidden_size)]]
param_losses = {}
completed_reps = {}
excess_capacity = {}
for hidden_size, sample_size in test_params:
    total_losses = []
    strike = False
    broke = False
    n = 0
    for n in range(n_reps):
        if False in excess_capacity.values():
            break
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
    excess_capacity[(hidden_size, sample_size)] = not broke
    completed_reps[(hidden_size, sample_size)] = n+1
    param_losses[(hidden_size, sample_size)] = total_losses

for params in param_losses.keys():
    print(f'{params}: {excess_capacity[params]}')
for params in param_losses.keys():
    for rep in range(completed_reps[params]):
        print(f'{params} rep {rep}: {param_losses[params][rep][-1]}')


if __name__ == "__main__":
    ...
