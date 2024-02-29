import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from Models.associator import Associator, Trainer

# Define input, hidden, and output sizes
# input_sizes = [1, 2, 3, 4, 5]
data_size = 4
context_size = 2
input_size = data_size + context_size
hidden_size = 5  # model capacity proxy
# output_sizes = [1, 2, 3, 4, 5]
output_size = data_size
sample_limit = 2*hidden_size
sample_sizes = range(sample_limit, sample_limit+5)


# Extra parameters
n_reps = 20
p = 0.05
epochs_ = -1
lr_ = 0.01

success_rate = np.zeros((len(sample_sizes),))
# success_rate[0] = n_reps


for idx, sample_size in enumerate(sample_sizes):
    print(f'\n---Training {hidden_size}-node model on {sample_size} {data_size}-D samples---')
    for n in range(n_reps):
        print(f'Repetition: {n+1}/{n_reps}')
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
        # Abort if more than one instance of memorization was unsuccessful
        if trainer.losses[-1] == 1.0:
            success_rate[idx] += 1

# From tally to frequency
success_rate = success_rate / n_reps
print(success_rate)

plt.plot(success_rate)
plt.xlabel('Sample Size')
plt.ylabel('Probability of Full Memorization')
plt.title('Probabilistic Capacity over sample sizes')
plt.show()

if __name__ == "__main__":
    ...
