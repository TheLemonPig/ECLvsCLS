import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt

from Models.associator import Associator, Trainer

# Define input, hidden, and output sizes
input_size = 5
hidden_size = 10  # model capacity proxy
output_size = 5

# Extra parameters
sample_size = 10  # data size proxy
epochs_ = 20000
lr_ = 0.01

# test_params = [(hidden_size, sample_size)]
test_params = [(10, 4*a) for a in range(2, 10)]
b_losses = []
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

    # Set B condition
    a_data[:, -1] = 0
    # Train the model on associating A with B
    trainer.train(a_data, b_data, num_epochs=epochs_, print_every=1000)
    b_losses.append(trainer.losses)

# ---- Present Results ----
epochs = np.arange(epochs_)
plt.figure(figsize=(10, 5))
top = max(test_params[-1])
for idx, run in enumerate(test_params):
    coef = run[1]/top
    plt.plot(epochs, b_losses[idx], label=f'{run[1]} samples against {run[0]} hidden nodes', c=(1-coef, 0, 0, 1))
plt.axhline(y=0.95, color='gray', linestyle='--', label='Memorization Threshold')
plt.title('Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    ...
