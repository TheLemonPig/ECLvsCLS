import matplotlib.pyplot as plt
import numpy as np

from utils import get_most_recent

# From tally to frequency
# success_rate = np.array([
#     1, 1, 0.9, 0.9, 0.75, 0.9, 0.45, 0.6, 0.5, 0.5, 0.65, 0.45, 0.35, 0.35, 0.4
# ])
# # 10 nodes: 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 1, 1, 0.95, 0.9, 0.85, 0.85, 0.75, 0.65, 0.7, 0.35, 0.85
# # 5 nodes: 1, 1, 0.9, 0.9, 0.75, 0.9, 0.45, 0.6, 0.5, 0.5
# # 10 nodes 2-dims: 1, 1, 1, 1, 1, 1, 1, 0.95, 0.9, 0.85, 0.9, 0.85, 0.9, 0.75, 0.8, 0.75, 0.7, 0.45, 0.4, 0.2, 0.3, 0.2, 0.15, 0
# plt.plot(success_rate)
# plt.xlabel('Sample Size')
# plt.ylabel('Probability of Full Memorization')
# plt.title('Probabilistic Capacity over sample sizes')
# plt.show()

# capacities = {(4, 4): {5: 5, 6: 8, 7: 8, 8: 11, 9: 16,
#                        10: 16, 11: 14, 12: 20, 13: 20, 14: 21,
#                        15: 18, 16: 20, 17: 21, 18: 22, 19: 27,
#                        20: 27}}
#
# for params in capacities.keys():
#     param_capacities = capacities[params]
#     capacity_array = np.array([[k, v] for k, v in param_capacities.items()])
#     plt.plot(capacity_array[:, 0], capacity_array[:, 1], label=f"Input/Output size: {params}")
# plt.legend()
# plt.xlabel('Nodes / Hidden Layer')
# plt.ylabel('Sample Size')
# plt.show()

# config = get_most_recent()
#
# first_epochs = max([int(config['first_epochs']/1000 * 10**(3-np.log10(hidden_size)))
#                     for hidden_size, _ in config['params']])

# top = np.log10(max(config['params'][-1]))
# for params in config['results']:
#     firsts, seconds = config['results'][params]
#
#     a_train_b_0_avg_0 = np.zeros((first_epochs,))
#     a_test_c_0_avg_0 = np.zeros((first_epochs,))
#     a_train_c_1_avg_1 = np.zeros((config['second_epochs'],))
#     a_train_b_0_avg_1 = np.zeros((config['second_epochs'],))
#     a_test_c_0_avg_1 = np.zeros((config['second_epochs'],))
#     for n in range(config['n_reps']):
#         first = firsts[n]
#         second = seconds[n]
#         a_train_b_0_avg_0 += np.array(first['train_accuracy']) / config['n_reps']
#         a_test_c_0_avg_0 += np.array(first['tests_accuracy'][0]) / config['n_reps']
#         a_train_c_1_avg_1 += np.array(second['train_accuracy']) / config['n_reps']
#         a_train_b_0_avg_1 += np.array(second['tests_accuracy'][0]) / config['n_reps']
#         a_test_c_0_avg_1 += np.array(second['tests_accuracy'][1]) / config['n_reps']
#
# for params in config['results']:
#     coef = np.log10(params[0]) / top
#     plt.plot(np.arange(len(a_train_b_0_avg_0)), a_train_b_0_avg_0, label=f'b=0 {params[1]} samples {params[0]} nodes',
#                   c=(0, 0, coef, 1))
#     plt.plot(np.arange(len(a_test_c_0_avg_0)), a_test_c_0_avg_0, label=f'c=0 {params[1]} samples {params[0]} nodes',
#                   c=(coef, 0, 0, 1))
#     plt.title('first: a_train on b=0 test c=0')
# plt.show()
# for params in config['results']:
#     coef = np.log10(params[0]) / top
#     plt.plot(np.arange(len(a_train_c_1_avg_1)), a_train_c_1_avg_1, label=f'c=1 {params[1]} samples {params[0]} nodes',
#                   c=(0, 0, coef, 1))
#     plt.plot(np.arange(len(a_train_b_0_avg_1)), a_train_b_0_avg_1, label=f'b=0 {params[1]} samples {params[0]} nodes',
#                   c=(coef, 0, 0, 1))
#     plt.title('second: a_train on c=1 test b=0')
# plt.show()
# for params in config['results']:
#     coef = np.log10(params[0]) / top
#     plt.plot(np.arange(len(a_train_c_1_avg_1)), a_train_c_1_avg_1, label=f'a_train_c_1_avg_1 {params[1]} samples {params[0]} nodes',
#                   c=(0, 0, coef, 1))
#     plt.plot(np.arange(len(a_test_c_0_avg_1)), a_test_c_0_avg_1, label=f'a_test_c_0_avg_1 {params[1]} samples {params[0]} nodes',
#                   c=(0, coef, 0, 1))
#     plt.title('second: a_train on c=1 test c=0')
# plt.show()
# for params in config['results']:
#     coef = np.log10(params[0]) / top
#     plt.plot(np.arange(len(a_train_c_1_avg_1)), a_train_c_1_avg_1, label=f'a_train_c_1_avg_1 {params[1]} samples {params[0]} nodes',
#                   c=(0, 0, coef, 1))
#     plt.plot(np.arange(len(a_train_b_0_avg_1)), a_train_b_0_avg_1, label=f'a_train_b_0_avg_1 {params[1]} samples {params[0]} nodes',
#                   c=(coef, 0, 0, 1))
#     plt.plot(np.arange(len(a_test_c_0_avg_1)), a_test_c_0_avg_1, label=f'a_test_c_0_avg_1 {params[1]} samples {params[0]} nodes',
#                   c=(0, coef, 0, 1))
#     plt.title('second: all three combined')
# plt.show()

import torch

# Define a simple neural network
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 3)
        self.fc2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the neural network
model = SimpleNet()

# Define input data
x = torch.tensor([1.0, 2.0])
x = torch.tensor(x, requires_grad=True)

# Forward pass
output = model(x)

# Define a dummy loss
loss = output.sum()

# Compute gradients
model.zero_grad()
loss.backward()

# Extract Jacobian matrix
jacobian = x.grad.view(1, -1)

print("Jacobian Matrix:")
print(jacobian)
