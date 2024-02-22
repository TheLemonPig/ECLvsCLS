import matplotlib.pyplot as plt
import numpy as np

capacities = {(4, 4): {5: 5, 6: 8, 7: 8, 8: 11, 9: 16,
                       10: 16, 11: 14, 12: 20, 13: 20, 14: 21,
                       15: 18, 16: 20, 17: 21, 18: 22, 19: 27,
                       20: 27}}

for params in capacities.keys():
    param_capacities = capacities[params]
    capacity_array = np.array([[k, v] for k, v in param_capacities.items()])
    plt.plot(capacity_array[:, 0], capacity_array[:, 1], label=f"Input/Output size: {params}")
plt.legend()
plt.xlabel('Nodes / Hidden Layer')
plt.ylabel('Sample Size')
plt.show()
