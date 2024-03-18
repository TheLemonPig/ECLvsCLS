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


def make_data(input_size, sample_size):
    # Define the range for each dimension
    dimension_range = (0, 1)

    # Create equally spaced points in each dimension
    points = np.linspace(dimension_range[0], dimension_range[1], sample_size)

    # Create a grid of all combinations of these points
    mesh = np.meshgrid(*[points] * input_size)

    # Reshape the grid to a 5D array where each row represents a point in 5D space
    points_5d = np.vstack([axis.flatten() for axis in mesh]).T

    # Print the 5D points
    print("5D Points:")
    print(points_5d)

    return torch.Tensor(points_5d)


if __name__ == "__main__":
    make_data(input_size=5, sample_size=10)
