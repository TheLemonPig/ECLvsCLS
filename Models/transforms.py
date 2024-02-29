# Data Transforms
import numpy as np
import torch


# Rotation in n_dims dimensions
def rotate(dataset, n_dims, axis=1):
    transform = torch.eye(dataset.shape[axis])  # identity
    for i in range(n_dims):
        transform[i, i] = 0
        transform[i, (i+1) % n_dims] = 1
    return torch.matmul(dataset, transform)


