import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np


# Define a simple neural network model
class Categorizer(nn.Module):
    def __init__(self):
        super(Categorizer, self).__init__()
        latent_dim = 10
        self.fc1 = nn.Linear(2, latent_dim)  # Input size: 2, Output size: 10
        self.norm = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, 1)  # Input size: 10, Output size: 1

    def forward(self, x):
        x = self.fc1(x)
        # x = self.norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, n_features=1):
        self.data = data
        self.n_features = n_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = self.data[idx]
        return batch[:self.n_features], batch[self.n_features:]


class Trainer:

    def __init__(self, model):
        self.model = model
        # Define a loss function and an optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)

    def train(self, data, batch_size=1, n_features=1, num_epochs=1000, test=None):
        data = torch.tensor(data, dtype=torch.float32)
        data = CustomDataset(data, n_features=n_features)
        losses = []
        test_results = []
        test_losses = []
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        test_loaders = []
        if test is not None:
            if type(test) == np.ndarray:
                test = [test]
            for dataset in test:
                dataset = torch.tensor(dataset, dtype=torch.float32)
                dataset = CustomDataset(dataset, n_features=n_features)
                test_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
                test_losses.append(list())
        for epoch in range(num_epochs):
            loss = None
            epoch_loss = 0
            for train_input, train_target in train_loader:
                self.optimizer.zero_grad()  # Zero the gradients
                output = self.model(train_input)  # Forward pass
                loss = self.criterion(output, train_target)  # Compute the loss
                epoch_loss += loss.item()
                # epoch_loss += ((output - train_target) ** 2 < 1).sum()
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights
            losses.append(epoch_loss/len(data.data))

            if test is not None and len(test) > 0:
                for idx, dataloader in enumerate(test_loaders):
                    test_loss = 0
                    n_samples = 0
                    for test_input, test_target in dataloader:
                        test_predict = self.model(test_input)  # Forward pass
                        test_loss += self.criterion(test_predict, test_target).item()  # Compute the loss
                        # test_loss += ((test_predict - test_target) ** 2 < 1).sum()
                        test_results.append(test_predict.detach().numpy())
                        n_samples += test_input.shape[0]
                    test_losses[idx].append(test_loss/n_samples)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        return losses, test_losses

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        predicted = self.model(x).detach().numpy()
        return predicted


if __name__ == '__main__':
    # Training data (pairs of numbers)
    X_train = np.array([[1.0, 2.0], [2.0, 3.0]])
    y_train = np.array([3.0, 5.0]).reshape((-1, 1))
    data_train = np.concatenate([X_train, y_train], axis=-1)

    mod = Categorizer()
    trainer = Trainer(mod)

    trainer.train(data_train, n_features=2, batch_size=2)
    # Test the model with new data
    X_test = torch.tensor([3.0, 4.0])

    predicted_output = trainer.predict(X_test)
    print(f'Input: {X_test}, Predicted Output: {predicted_output.item()}')
