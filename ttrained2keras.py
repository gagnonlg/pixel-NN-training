import argparse
import os.path
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import ROOT

# Note on input order: comparing the output of genconfig.py and
# e.g. https://svnweb.cern.ch/cern/wsvn/atlas-rjansky/AGILEPack/trunk/config_WeightsPosition1.yaml
# parameters seem to be in the same order

ROOT.gROOT.SetBatch()
scriptdir = os.path.dirname(os.path.abspath(__file__))
ROOT.gROOT.ProcessLine('.L %s/TTrainedNetwork.so' % scriptdir)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in', required=True, dest='inp')
    p.add_argument('--out', required=True)
    return p.parse_args()

def load_net(path):
    tfile = ROOT.TFile(path, 'READ')
    return tfile.Get('TTrainedNetwork')

def get_weights(net):
    weights = []
    for matrix in net.weightMatrices():
        weights.append(np.zeros((matrix.GetNrows(),matrix.GetNcols())))
        for i in range(matrix.GetNrows()):
            for j in range(matrix.GetNcols()):
                weights[-1][i,j] = matrix(i,j)
    return weights

def get_thresholds(net):
    thresholds = []
    for vect in net.getThresholdVectors():
        thresholds.append(np.zeros(vect.GetNrows()))
        for i in range(vect.GetNrows()):
            thresholds[-1][i] = vect(i)
    return thresholds

def get_struct(net):
    struct =  [net.getnInput()]
    struct += list(net.getnHiddenLayerSize())
    struct += [net.getnOutput()]
    return struct

def get_normalization(net):
    stats = net.getInputs()
    norm = np.zeros((2,stats.size()))
    for i,s in enumerate(stats):
        norm[0][i] = -s.offset
        norm[1][i] = 1.0/s.scale
    return norm

def build_model(struct, weights, thresholds, is_regression):
    model = Sequential()
    for i in range(1, len(struct)):
        model.add(Dense(input_dim=struct[i-1], output_dim=struct[i]))
        model.layers[-1].set_weights([weights[i-1], thresholds[i-1]])
        if i < (len(struct) - 1):
            model.add(Activation('sigmoid'))
        else:
            act = 'linear' if is_regression else 'softmax'
            model.add(Activation(act))

    loss = 'binary_crossentropy' if act == 'softmax' else 'mae'
    model.compile(SGD(),loss)
    return model

def get_model(path):
    net = load_net(path)
    struct = get_struct(net)
    w = get_weights(net)
    b = get_thresholds(net)
    model = build_model(struct, w, b, net.getIfLinearOutput())
    norm = get_normalization(net)
    return model, norm

def save_model(model, norm, output):
    with open(output + '.model.yaml', 'w') as yfile:
        yfile.write(model.to_yaml())
    np.savetxt(output + '.normalization.txt', norm)
    model.save_weights(output + '.weights.hdf5', overwrite=True)

if __name__ == '__main__':
    args = get_args()
    model, norm = get_model(args.inp)
    save_model(model, norm, args.out)

