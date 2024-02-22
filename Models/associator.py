import torch.nn as nn
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from tqdm import tqdm


class Associator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, context_size):
        super(Associator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.context_size = context_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.losses = []

    def train(self, input_data, target_data, test_data=None, test_condition=0,
              num_epochs=1000, print_every=100):
        if num_epochs == -1:
            delta_break = True
            num_epochs = int(10e6)
        else:
            delta_break = False
        continuous_loss = []
        self.losses = []
        test_losses = []
        test_input = deepcopy(input_data)
        if self.model.context_size > 0:
            test_input[:, -self.model.context_size:] = 0
            if test_condition == 0:
                test_input[:, -self.model.context_size:-self.model.context_size//2] = 1
            elif test_condition == 1:
                test_input[:, -self.model.context_size//2:] = 1
        loss_val = None
        acc_val = 0
        progress_bar = tqdm(total=num_epochs, desc='Training')
        for epoch in range(num_epochs):
            # Forward pass
            output = self.model(input_data)

            # Compute the loss
            loss = self.criterion(output, target_data)
            continuous_loss.append(loss.item())
            accuracy = classify_associations(output.detach(), target_data)
            self.losses.append(accuracy.item())

            if test_data is not None:
                output = self.model(test_input)
                test_losses.append(classify_associations(output.detach(), test_data).item())

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            acc_val = accuracy.item()
            progress_bar.update(1)
            progress_bar.set_postfix_str(f'Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%')
            # f'Loss: {loss_val:.4f}, Accuracy: {acc_val*100:.2f}%'
            # Print the loss every 'print_every' epochs
            #if (epoch + 1) % print_every == 0:
            #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%')
            if delta_break and epoch > 10000:
                delta = continuous_loss[-10000] - continuous_loss[-1]
                if delta < 10e-5:
                    print(f'\nfrom {continuous_loss[-10000]:.6f} to {continuous_loss[-1]:.6f} over 10e4 epochs')
                    print(f'{delta:.6f} < 10e-5 and hence by arbitrary threshold, training has converged')
                    break
                elif sum(self.losses[-5:]) == 5.0:
                    print("\n100% for 5 trials in a row!! Training Complete")
                    break

        return test_losses


def classify_associations(output, target_data):
    classified = nearest_neighbors(output, target_data)
    return classified.sum()/len(classified)


def nearest_neighbors(output_vectors, target_vectors):

    # Calculate cosine similarities
    similarities = F.cosine_similarity(output_vectors.unsqueeze(1), target_vectors, dim=2)

    # Find the indices of the k nearest neighbors
    _, indices = torch.topk(similarities, k=1, dim=1, largest=True)

    return indices.flatten() == torch.arange(indices.shape[0])

# def nearest_neighbors(vector, vectors, k=1):
#     # Convert vectors to numpy arrays if not already
#     vector = np.array(vector).reshape(1, -1)
#     vectors = np.array(vectors)
#
#     # Calculate cosine similarities
#     similarities = cosine_similarity(vector, vectors)
#
#     # Find the indices of the k nearest neighbors
#     indices = np.argsort(similarities.flatten())[::-1][:k]
#
#     return indices
#
#
# def classify_associations(output, target_data):
#     loss = 0
#     for idx, sample in enumerate(output):
#         loss += (nearest_neighbors(sample, target_data) == idx)
#     return loss / output.shape[0]
