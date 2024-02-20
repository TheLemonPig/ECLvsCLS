import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt

from Models.associator import Associator, Trainer

# Define input, hidden, and output sizes
input_size = 5
# hidden_size = 10  # model capacity proxy
output_size = 5

# Extra parameters
sample_size = 10  # data size proxy
epochs_ = 40000
lr_ = 0.01

# test_params = [(hidden_size, sample_size)]
test_params = [(2*a, sample_size) for a in range(1, 11)]
b_losses = []
c_losses = []
for hidden_size, sample_size in test_params:
    # Create an instance of the VectorMapper class
    model = Associator(input_size, hidden_size, output_size)

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
    a_data[:, -1] = 0
    # Train the model on associating A with B
    trainer.train(a_data, b_data, num_epochs=epochs_, print_every=1000)
    # b_losses.append(trainer.losses)

    # Set C condition
    a_data[:, -1] = 1
    # Train the model on associating A with C while testing on B
    test_losses = trainer.train(a_data, c_data, b_data, num_epochs=epochs_, print_every=1000)
    b_losses.append(test_losses)
    c_losses.append(trainer.losses)


# ---- Present Results ----
epochs = np.arange(epochs_)
n_rows = 2
n_cols = 5
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4, 4))
for idx, run in enumerate(test_params):
    row = idx // 5
    col = idx % 5
    axs[row, col].plot(epochs, c_losses[idx], color='black')
    axs[row, col].plot(epochs, b_losses[idx], color='red')
    axs[row, col].set_title(f'{run[1]} samples {run[0]} nodes')
    axs[row, col].axhline(y=1, color='gray', linestyle='--')
    # axs[row, col].set_xlabel('Epoch')
    # axs[row, col].set_ylabel('Accuracy')
fig.suptitle('Classification Accuracy')
fig.tight_layout()
plt.show()

if __name__ == "__main__":
    ...
