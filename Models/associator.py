import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time import sleep


class Associator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_conditions=2, hidden_layers=1):
        super(Associator, self).__init__()
        self.fc1 = nn.Linear(input_size+n_conditions, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers-1)]
        # TODO: Test variable layer feature - speed and replicable against current setup
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.n_conditions = n_conditions

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # for layer in self.layers:
        #     x = layer(x)
        #     x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.losses = dict()

    def train(self, training_data, tests=None,
              num_epochs=1000, stops=('epochs', ), epoch_min=10000, delta_min=10e-5, normalize=False, sleep_time=0):
        # Set training regime
        regime = dict({'epochs': False,
                       'delta_train': False, 'one_delta_test': False, 'all_delta_test': False,
                       'train_accuracy': False, 'one_test_accuracy': False, 'all_test_accuracy': False})

        # Check that stop arg is a valid training regime
        for stop in stops:
            if stop not in regime.keys():
                raise KeyError('Unrecognized training regime')
            else:
                regime[stop] = True

        # remove epochs as a meaningful training parameter when epochs is not the training regime
        if not regime['epochs']:
            # we check at the end of training to make sure this wasn't reached
            num_epochs = int(10e6)

        # Complete training data formatting
        train_input, train_output, train_condition = training_data
        train_input.requires_grad_(True)
        condition_train = torch.zeros((train_input.shape[0], self.model.n_conditions))
        if train_condition is not None:
            condition_train[:, train_condition] = 1
        else:
            raise UserWarning('Training data supplied without training condition')
        train_input = torch.cat((train_input, condition_train), dim=1)

        # Complete test data formatting if relevant
        n_tests = 0
        test_inputs = []
        test_outputs = []
        if tests is not None:
            # Make sure all test_condition values exist and are valid values
            n_tests = len(tests)
            for i in range(n_tests):
                test_input, test_output, test_condition = tests[i]
                condition_test = torch.zeros((test_input.shape[0], self.model.n_conditions))
                condition_test[:, test_condition] = 1
                test_input = torch.cat((test_input, condition_test), dim=1)
                test_inputs.append(test_input)
                test_outputs.append(test_output)

        # creating / reinitializing dictionary of losses
        self.losses = {'train_continuous': [], 'train_accuracy': [],
                       'tests_continuous': [[] for _ in range(n_tests)], 'tests_accuracy': [[] for _ in range(n_tests)]}
        # allow print enough time to catch up to tqdm to prevent overlap
        sleep(sleep_time)
        # initialize progress bar
        progress_bar = tqdm(total=num_epochs, desc=f'Regime: {stops}')

        # begin training loop
        epoch = 0
        for epoch in range(num_epochs):

            # forward pass
            train_predict = self.model(train_input)
            train_loss = self.criterion(train_predict, train_output)

            # collect training losses
            self.losses['train_continuous'].append(train_loss.item())
            train_accuracy = classify_associations(train_predict.detach(), train_output, normalize)
            self.losses['train_accuracy'].append(train_accuracy.item())

            # Backward pass over training loss and update weights
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            # Update progress bar with training statistics for this epoch
            loss = train_loss.item()
            accuracy = train_accuracy.item()
            progress_bar.update(1)  # increment epoch counter
            progress_bar.set_postfix_str(f'Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%')

            # run all tests
            self.model.eval()
            with torch.no_grad():
                for i in range(n_tests):
                    # forward pass
                    test_predict = self.model(test_inputs[i])
                    test_loss = self.criterion(test_predict, test_outputs[i])

                    # collect test losses
                    self.losses['tests_continuous'][i].append(test_loss.item())
                    test_accuracy = classify_associations(test_predict.detach(), test_outputs[i], normalize)
                    self.losses['tests_accuracy'][i].append(test_accuracy.item())
            self.model.train()

            # Identify whether training should be curtailed based on our training regime
            if regime['epochs']:
                continue
            if regime['train_accuracy']:
                if sum(self.losses['train_accuracy'][-1:]) == 1.0:
                    print("\n100% achieved!! Training Complete")
                    sleep(sleep_time)
                    break
            if regime['one_test_accuracy'] or regime['all_test_accuracy']:
                delta_break = False
                for i in range(n_tests):
                    if sum(self.losses['test_accuracy'][i][-1:]) == 1.0:
                        if regime['one_test_accuracy']:
                            delta_break = True or delta_break
                        if regime['all_test_accuracy']:
                            delta_break = True and delta_break
                if delta_break:
                    if regime['one_test_accuracy']:
                        print("\n100% achieved on one test!! Testing Complete")
                    elif regime['all_test_accuracy']:
                        print("\n100% achieved on all tests!! Testing Complete")
                    sleep(sleep_time)
                    break
            if regime['delta_train'] and epoch > epoch_min:
                loss: list = self.losses['train_continuous']
                past_loss: float = loss[-epoch_min]
                recent_loss: float = loss[-1]
                delta = past_loss - recent_loss
                if delta < delta_min:
                    print(f'\nfrom {past_loss:.6f} to {recent_loss:.6f} over 10e4 epochs')
                    print(f'{delta:.6f} < {delta_min} and hence by arbitrary threshold, training has converged')
                    sleep(sleep_time)
                    break
            if (regime['one_delta_test'] or regime['all_delta_test']) and epoch > epoch_min:
                delta_break = False
                for i in range(n_tests):
                    past_loss = self.losses['test_continuous'][i][-epoch_min]
                    recent_loss = self.losses['test_continuous'][i][-1]
                    delta = past_loss - recent_loss
                    if delta < delta_min:
                        if regime['one_delta_test']:
                            delta_break = True or delta_break
                        elif regime['all_delta_test']:
                            delta_break = True and delta_break
                        if delta_break:
                            if regime['one_test_accuracy']:
                                print(f'delta < {delta_min} and hence by arbitrary threshold, testing has converged')
                            sleep(sleep_time)
                            break
        # check that epoch limit is not reached if the regime is not epochs
        if not regime['epochs'] and epoch == num_epochs:
            raise RuntimeWarning('Upper Limit of Epochs reached while training in a non-epoch regime')
        self.losses['n_epochs'] = (epoch+1, num_epochs)
        if epoch + 1 < num_epochs:
            self.pad_results()
        return self.losses

    def pad_results(self):
        completed_epochs, planned_epochs = self.losses['n_epochs']
        num_padding = planned_epochs - completed_epochs
        self.losses['train_continuous'] += [self.losses['train_continuous'][-1] for _ in range(num_padding)]
        self.losses['train_accuracy'] += [self.losses['train_accuracy'][-1] for _ in range(num_padding)]
        for i in range(len(self.losses['tests_accuracy'])):
            self.losses['tests_continuous'][i] += [self.losses['tests_continuous'][i][-1] for _ in range(num_padding)]
            self.losses['tests_accuracy'][i] += [self.losses['tests_accuracy'][i][-1] for _ in range(num_padding)]


# These function are used to convert continuous predictions into classification predictions
def classify_associations(output, target_data, normalize=False):
    classified = nearest_neighbors(output.unsqueeze(1), target_data, normalize=normalize)
    return classified.sum()/len(classified)


# Classification is based on cosine similarity of vectors
def nearest_neighbors(output_vectors, target_vectors, normalize=True):
    # normalize inputs if you want to take relative magnitude into account
    if normalize:
        output_vectors = F.normalize(output_vectors, p=2, dim=1)
        target_vectors = F.normalize(target_vectors, p=2, dim=1)
    # Calculate cosine similarities
    similarities = F.cosine_similarity(output_vectors, target_vectors, dim=2)
    # Find the indices of the k nearest neighbors
    _, indices = torch.topk(similarities, k=1, dim=1, largest=True)
    return indices.flatten() == torch.arange(indices.shape[0])
