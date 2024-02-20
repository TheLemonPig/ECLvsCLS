import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt

from Models.associator import Associator, Trainer

# Example usage:
# Define input, hidden, and output sizes
input_size = 5
hidden_size = 10
output_size = 5

# Extra parameters
sample_size = 20
epochs_ = 10000
lr_ = 0.01

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
b_losses = trainer.losses

# Set C condition
a_data[:, -1] = 1
# Train the model on associating A with C, with testing A on C
test_losses = trainer.train(a_data, c_data, b_data, num_epochs=epochs_, print_every=1000)
c_losses = trainer.losses

# Test the trained model
output = model(a_data[0])
print(f'Test Input: {a_data}')
print(f'Predicted Output: {output}')
print(f'Target Output: {b_data[0]}')

# ---- Present Results ----
epochs = np.arange(epochs_)
plt.figure(figsize=(10, 5))
plt.plot(epochs, c_losses, label='C Test Loss', color='black')
plt.plot(epochs, test_losses, label='B 2nd Test Loss', color='orange')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    ...
