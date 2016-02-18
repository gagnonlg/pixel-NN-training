import csv
from subprocess import check_output
import numpy as np

def load_data(path,targets,shape=None):
    if shape is not None:
        data = np.zeros(shape)
        header = None
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            fst = True
            for i,row in enumerate(reader):
                if fst:
                    header = row
                    fst = False
                    continue
                data[i-1,:] = np.array(row)
    else:
        data = np.loadtxt(path,delimiter=',',skiprows=1)

    x = data[:,:-targets]
    y = data[:,-targets:]

    return x,y,header

def load_generator(path, batch, i_inputs, i_targets, norm):
    dataX = np.zeros((batch, ninputs))
    dataY = np.zeros((batch, ntargets))
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n = 0
        while True:
            for i,row in enumerate(reader):
                if i == 0:
                    continue
                line = np.array(row)
                dataX[n] = (line[i_inputs] - norm[i_inputs,0]) / norm[i_inputs,1]
                dataY[n] = (line[i_targets] - norm[i_targets,0]) / norm[i_targets,1]
                n += 1
                if n == batch:
                    n = 0
                    yield (dataX, dataY)


def get_shape(path, skiprows=0):
    nrow = int(check_output(['wc', '-l', path]).split()[0]) - skiprows
    ncol = int(check_output("head -1 %s | awk -F, '{print NF}'" % path, shell=True))
    return nrow, ncol

def get_header(path):
    with open(path, 'r') as f:
        return f.readline()[:-1].split(',')

def load_data_bulk(path, shape, extra=0):
    nrow = shape[0]
    ncol = shape[1]
    data = np.zeros((nrow, ncol + extra))
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(reader):
            if i == 0:
                continue
            data[i-1, 0:ncol] = np.array(row)

    return data

def get_data_config_names(path, meta):
    do_inputs = False
    do_metadata = False
    do_targets = False
    inputs = []
    targets = []
    metadata = []
    with open(path, 'r') as f:
        for line in f:
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

def get_data_config_indices(inputs, targets, metadata, header):
    i_inputs = [i for i,col in enumerate(header) if col in inputs]
    i_targets = [i for i,col in enumerate(header) if col in targets]
    i_metadata = [i for i,col in enumerate(header) if col in metadata]
    return i_inputs, i_targets, i_metadata

def get_data_config(path, header, meta=True):
    i,t,m = get_data_config_names(path, meta)
    ii,it,im = get_data_config_indices(i,t,m,header)
    if meta:
        return ii, it, im
    else:
        return ii, it

