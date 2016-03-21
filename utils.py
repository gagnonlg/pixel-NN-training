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


def save_scale_offset(norm, output):
    scale = norm['scale']
    offset = norm['offset']
    out = np.empty((2, scale.shape[0]))
    out[0] = scale
    out[1] = offset
    np.savetxt(output + '.scale_offset.txt', out)


def load_scale_offset(path):
    inp = np.loadtxt(path)
    return {'scale': inp[0], 'offset': inp[1]}
