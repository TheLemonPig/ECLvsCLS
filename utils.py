import pickle
import os
import torch


def get_most_recent(prefix='', script_path='..', config=None, ignore=None):
    most_recent = None
    ignore_args = [] if ignore is None else ignore
    ignore_args += ['stops', 'second_epochs', 'first_epochs', 'fit_samples', 'fit_models', 'seeds',
                    'model_seed_capacities', 'max_epochs_needed', 'noisy_generalization']
    log_path = os.path.join(script_path, 'Logs')
    for file in os.listdir(log_path):
        if file.startswith(prefix):
            match = True
            if config is not None:
                with open(os.path.join(log_path, file), 'rb') as f:
                    temp_log = pickle.load(f)
                for arg in config.keys():
                    if arg not in ignore_args:
                        try:
                            match = match and (arg in config.keys()) and (temp_log[arg] == config[arg])
                        except KeyError:
                            match = False
            if match:
                if most_recent is None or most_recent < file:
                    most_recent = file
    if most_recent is None:
        raise RuntimeError('No log files to retrieve')
    else:
        with open(os.path.join(log_path, most_recent), 'rb') as f:
            log = pickle.load(f)
        return log


def make_data(sample_size, input_size, seed):
    torch.manual_seed(seed)
    return torch.randn(sample_size, input_size)


if __name__ == '__main__':
    log_test = get_most_recent()
    print(log_test)


