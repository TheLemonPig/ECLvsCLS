import torch
import torch.nn.functional as F


def nearest_neighbors(input_vectors, target_vectors):
    """
    Find the indices of the k nearest neighbors for each vector among a set of other vectors.

    Parameters:
    - vectors (torch.Tensor): Input vectors for which nearest neighbors are to be found.
    - other_vectors (torch.Tensor): A batch of vectors among which nearest neighbors are to be found.
    - k (int): The number of nearest neighbors to find.

    Returns:
    - indices (torch.Tensor): Indices of the k nearest neighbors for each input vector.
    """

    # Calculate cosine similarities
    similarities = F.cosine_similarity(input_vectors.unsqueeze(1), target_vectors, dim=2)

    # Find the indices of the k nearest neighbors
    _, indices = torch.topk(similarities, k=1, dim=1, largest=True)

    return indices.flatten() == torch.arange(indices.shape[0])

# Example usage:
# Define an example batch of vectors and another batch of vectors
batch_vectors = torch.randn(5, 10)  # 5 vectors of dimension 10
other_vectors = torch.randn(7, 10)  # 7 vectors of dimension 10

# Find the nearest neighbors for each vector in the batch
nearest_indices = nearest_neighbors(batch_vectors, other_vectors)

print("Batch Vectors:")
print(batch_vectors)

print("\nOther Vectors:")
print(other_vectors)

print("\nIndices of the Nearest Neighbors for Each Vector:")
print(nearest_indices)


