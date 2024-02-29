import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt

from Models.associator import Associator, Trainer

# Define input, hidden, and output sizes
data_size = 5
context_size = 2
assert context_size % 2 == 0
input_size = data_size + context_size
hidden_size = 10000  # model capacity proxy
output_size = data_size

# Extra parameters
sample_size = 10  # data size proxy
epochs_ = 2000
lr_ = 0.01
n_reps = 20

# test_params = [(hidden_size, sample_size)]
# test_params = [(10**a, sample_size, epochs_*4**(3-a)) for a in range(1, 4)]
test_params = [(a, sample_size) for a in [5, 12, 100, 1000]]
b_losses = []
c_losses = []
for hidden_size, sample_size, epochs in test_params:
    total_test_losses = np.zeros((epochs_,))
    total_trainer_losses = np.zeros((epochs_,))
    for n in range(n_reps):
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

        # Random C data
        c_data = torch.randn(sample_size, output_size)

        # Set B condition
        if context_size > 0:
            a_data[:, -context_size:] = 0
            a_data[:, -context_size:-context_size // 2] = 1
        # Train the model on associating A with B
        trainer.train(a_data, b_data, num_epochs=epochs, print_every=100)

        # Set C condition
        if context_size > 0:
            a_data[:, -context_size:] = 0
            a_data[:, -context_size//2:] = 1
        # Train the model on associating A with C while testing on B
        test_losses = trainer.train(a_data, c_data, b_data, test_condition=0, num_epochs=epochs_, print_every=100)
        total_test_losses += np.array(test_losses) / n_reps
        total_trainer_losses += np.array(trainer.losses) / n_reps
    b_losses.append(total_test_losses)
    c_losses.append(total_trainer_losses)


# ---- Present Results ----
epochs = np.arange(epochs_)
plt.figure(figsize=(10, 5))
top = np.log10(max(test_params[-1]))
for idx, run in enumerate(test_params):
    coef = np.log10(run[0]) / top
    plt.plot(epochs, c_losses[idx], label=f'C Training {run[1]} samples {run[0]} nodes', c=(0, 0, coef, 1))
    plt.plot(epochs, b_losses[idx], label=f'B Testing {run[1]} samples {run[0]} nodes', c=(coef, 0, 0, 1))
plt.axhline(y=1, color='gray', linestyle='--', label='100% Memorization')
plt.title('Average Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    ...
