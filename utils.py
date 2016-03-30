import numpy as np


def get_data_config_names(path, meta):
    do_inputs = False
    do_metadata = False
    do_targets = False
    inputs = []
    targets = []
    metadata = []
    with open(path, 'r') as cfg:
        for line in cfg:
            if line.strip('\n') == "inputs:":
                do_inputs = True
                do_metadata = False
                do_targets = False
                continue
            if line.strip('\n') == "metadata:":
                do_inputs = False
                do_metadata = True
                do_targets = False
                continue
            if line.strip('\n') == "targets:":
                do_inputs = False
                do_metadata = False
                do_targets = True
                continue
            if not line.startswith('  - '):
                continue
            if do_inputs:
                inputs.append(line.strip(' -\n'))
            elif do_metadata and meta:
                metadata.append(line.strip(' -\n'))
            elif do_targets:
                targets.append(line.strip(' -\n'))

    return inputs, targets, metadata


def save_normalization(norm, output):
    mean = norm['mean']
    std = norm['std']
    out = np.empty((2, std.shape[0]))
    out[0] = mean
    out[1] = std
    np.savetxt(output + '.normalization.txt', out)


def load_normalization(path):
    inp = np.loadtxt(path)
    return {'mean': inp[0], 'std': inp[1]}

