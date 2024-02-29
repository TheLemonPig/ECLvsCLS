import matplotlib.pyplot as plt
import numpy as np

# From tally to frequency
success_rate = np.array([
    1, 1, 0.9, 0.9, 0.75, 0.9, 0.45, 0.6, 0.5, 0.5, 0.65, 0.45, 0.35, 0.35, 0.4
])
# 10 nodes: 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 1, 1, 0.95, 0.9, 0.85, 0.85, 0.75, 0.65, 0.7, 0.35, 0.85
# 5 nodes: 1, 1, 0.9, 0.9, 0.75, 0.9, 0.45, 0.6, 0.5, 0.5
# 10 nodes 2-dims: 1, 1, 1, 1, 1, 1, 1, 0.95, 0.9, 0.85, 0.9, 0.85, 0.9, 0.75, 0.8, 0.75, 0.7, 0.45, 0.4, 0.2, 0.3, 0.2, 0.15, 0
plt.plot(success_rate)
plt.xlabel('Sample Size')
plt.ylabel('Probability of Full Memorization')
plt.title('Probabilistic Capacity over sample sizes')
plt.show()

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
