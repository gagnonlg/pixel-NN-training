import itertools as it
import json
import os

from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import ROOT

__all__ = [
    'init',
    'from_keras',
    'to_keras'
]


def init():
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    sopath = '{0}/TTrainedNetwork.so'.format(scriptdir)
    if os.path.isfile(sopath):
        ROOT.gROOT.ProcessLine('.L %s/TTrainedNetwork.so' % scriptdir)
    else:
        raise RuntimeError('{0}: file not found'.format(sopath))


def from_keras(model, normalization=None):

    config = json.loads(model.to_json())

    nInput = config['layers'][0]['input_dim']
    nHidden = len([l for l in config['layers'] if l['name'] == 'Dense']) - 1
    nOutput = config['layers'][-2]['output_dim']
    nHiddenLayerSize = _get_hidden_layer_size(config)
    thresholdVectors = _get_thresholds(model)
    weightMatrices = _get_weights(model)
    activationFunction = 1 # sigmoid2
    linearOutput = _get_linear_output(config)
    normalizeOutput = normalization is not None

    ttrained = ROOT.TTrainedNetwork(
        nInput,
        nHidden,
        nOutput,
        nHiddenLayerSize,
        thresholdVectors,
        weightMatrices,
        activationFunction,
        linearOutput,
        normalizeOutput
    )

    if normalization is not None:
        _set_normalization(ttrained, normalization)

    return ttrained


def _set_normalization(ttrained, normalization):

    offsets = ROOT.vector('double')()
    for u in np.nditer(normalization['mean']):
        offsets.push_back(-u)
    ttrained.setOffsets(offsets)

    scales = ROOT.vector('double')()
    for s in np.nditer(normalization['std']):
        scales.push_back(1.0/s)
    ttrained.setScales(scales)


def _get_linear_output(config):
    pairs = [('activation','linear') in l.items() for l in config['layers']]
    return pairs[-1]

def _get_hidden_layer_size(config):
    vect = ROOT.vector('Int_t')()
    for hl in [l for l in config['layers'] if l['name'] == 'Dense'][1:]:
        vect.push_back(hl['input_dim'])
    return vect


def _get_weights(model):
    weights = ROOT.vector('TMatrixT<double>*,allocator<TMatrixT<double>*> ')()
    for i, layer in enumerate(model.layers):
        if type(layer) == Dense:
            weights.push_back(_array_to_tmatrix(layer.get_weights()[0]))
    return weights


def _get_thresholds(model):
    thresh = ROOT.vector('TVectorT<double>*,allocator<TVectorT<double>*> ')()
    for i, layer in enumerate(model.layers):
        if type(layer) == Dense:
            thresh.push_back(_array_to_tvector(layer.get_weights()[1]))
    return thresh


def _array_to_tmatrix(arr):
    tmat = ROOT.TMatrixD(arr.shape[0], arr.shape[1])
    for i, j in it.product(range(arr.shape[0]), range(arr.shape[1])):
        tmat[i][j] = arr[i,j]

    return tmat


def _array_to_tvector(arr):
    tvec = ROOT.TVectorD(arr.shape[0])
    for i in range(arr.shape[0]):
        tvec[i] = arr[i]

    return tvec


def to_keras(ttrained):

    model = _build_model(
        struct=_tt_get_struct(ttrained),
        weights=_tt_get_weights(ttrained),
        thresholds=_tt_get_thresholds(ttrained),
        is_regression=ttrained.getIfLinearOutput()
    )
    norm = _tt_get_normalization(ttrained)

    return model, norm

def _build_model(struct, weights, thresholds, is_regression):

    def sigmoid2(x):
        return K.sigmoid(2*x)

    model = Sequential()

    for i in range(1, len(struct)):
        model.add(Dense(input_dim=struct[i-1], output_dim=struct[i]))
        model.layers[-1].set_weights([weights[i-1], thresholds[i-1]])

        if i < (len(struct) - 1):
            model.add(Activation(sigmoid2))
        else:
            act = 'linear' if is_regression else 'softmax'
            model.add(Activation(act))

    loss = 'binary_crossentropy' if act == 'softmax' else 'mae'
    model.compile(SGD(), loss)

    return model


def _tt_get_struct(ttrained):
    struct =  [ttrained.getnInput()]
    struct += list(ttrained.getnHiddenLayerSize())
    struct += [ttrained.getnOutput()]
    return struct


def _tt_get_weights(ttrained):
    weights = []
    for matrix in ttrained.weightMatrices():
        weights.append(np.zeros((matrix.GetNrows(), matrix.GetNcols())))
        for i in range(matrix.GetNrows()):
            for j in range(matrix.GetNcols()):
                weights[-1][i,j] = matrix(i, j)
    return weights


def _tt_get_thresholds(ttrained):
    thresholds = []
    for vect in ttrained.getThresholdVectors():
        thresholds.append(np.zeros(vect.GetNrows()))
        for i in range(vect.GetNrows()):
            thresholds[-1][i] = vect(i)
    return thresholds


def _tt_get_normalization(ttrained):

    stats = ttrained.getInputs()
    norm = {
        'mean' : np.zeros(stats.size()),
        'std' : np.ones(stats.size()),
    }

    for i,s in enumerate(stats):
        norm['mean'][i] = -s.offset
        norm['std'][i] = 1.0/s.scale

    return norm
