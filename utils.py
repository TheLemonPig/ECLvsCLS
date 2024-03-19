import pickle
import os


def get_most_recent(prefix=''):
    most_recent = None
    file = None
    for file in os.listdir('Logs'):
        if file.startswith(prefix):
            if most_recent is None:
                most_recent = file
            else:
                if most_recent < file:
                    most_recent = file
    if file is None:
        raise RuntimeError('No log files to retrieve')
    else:
        with open(os.path.join('Logs', file), 'rb') as f:
            log = pickle.load(f)
        return log


if __name__ == '__main__':
    log_test = get_most_recent()
    print(log_test)
