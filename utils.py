import pickle
import os
import torch


def get_most_recent(prefix='', script_path='..', config=None):
    most_recent = None
    log_path = os.path.join(script_path, 'Logs')
    for file in os.listdir(log_path):
        if file.startswith(prefix):
            if most_recent is None:
                most_recent = file
            else:
                match = True
                if config is not None:
                    with open(os.path.join(log_path, most_recent), 'rb') as f:
                        temp_log = pickle.load(f)
                    for arg in temp_log.keys():
                        match = match and (arg in config.keys()) and (temp_log[arg] == config[arg])
                if match:
                    if most_recent < file:
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


