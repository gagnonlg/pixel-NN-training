import argparse
import logging
import os.path
import re
from keras.layers.core import Dense, Activation
from keras.models import Sequential, model_from_yaml
from keras.optimizers import SGD
import numpy as np
import ROOT

# Note on input order: comparing the output of genconfig.py and
# e.g. https://svnweb.cern.ch/cern/wsvn/atlas-rjansky/AGILEPack/trunk/config_WeightsPosition1.yaml
# parameters seem to be in the same order

logging.basicConfig(level=logging.DEBUG)

ROOT.gROOT.SetBatch()
scriptdir = os.path.dirname(os.path.abspath(__file__))

logging.info('loading TTrainedNetwork.so')
ROOT.gROOT.ProcessLine('.L %s/TTrainedNetwork.so' % scriptdir)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--normalization', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args()

def load_net(modelp, weights, normp):

    model = model_from_yaml(open(modelp, 'r').read())
    model.load_weights(weights)
    norm = np.loadtxt(normp) if (normp is not None) else None
    return model, norm

def build_ttrained(model, norm):

    nInput = None
    nHidden = 0
    nOutput = None

    activationFunction = None

    nHiddenLayerSize = ROOT.vector('Int_t')()

    weights = []
    thresholds = []

    for n,layer in enumerate(model.layers):
        if type(layer) == Dense:

            W = layer.get_weights()[0]
            weights.append(ROOT.TMatrixD(W.shape[1], W.shape[0]))
            logging.debug('Adding layer %d weights with shape (%d,%d)' %
                          (len(weights), W.shape[0], W.shape[1]))
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    weights[-1][j][i] = W[i,j]

            b = layer.get_weights()[1]
            thresholds.append(ROOT.TVectorD(b.shape[0]))
            logging.debug('Adding layer %d thresholds with shape (%d,)' %
                          (len(weights), b.shape[0]))
            for i in range(b.shape[0]):
                thresholds[-1][i] = b[i]

            nHiddenLayerSize.push_back(W.shape[1])

            if n == 0:
                nInput = W.shape[0]
                logging.debug('setting nInput to %d' % nInput)

            if n == len(model.layers) - 2:
                nOutput = W.shape[1]
                logging.debug('setting nOutput to %d' % nOutput)
            else:
                nHidden += 1
                logging.debug('incrementing nHidden to %d', nHidden)
                logging.debug('Registering hidden layer %d size of %d' %
                              (len(weights), W.shape[1]))


        if type(layer) == Activation:
            actf = re.match('<function (.+) at .+>', str(layer.activation))
            if actf is None:
                logging.error('unable to parse activation function')
                exit(1)

            if n < len(model.layers) - 2:
                if actf.group(1) == 'sigmoid':
                    logging.debug('setting activation to sigmoid')
                    logging.warning('found "sigmoid" activation but TTrainedNetwork implements "sigmoid2"')
                    activationFunction = 1
                elif actf.group(1) == 'sigmoid2':
                    logging.debug('setting activation to sigmoid2')
                    activationFunction = 1
                else:
                    logging.error('unrecognized activation function: %s' % actf.group(1))
            else:
                linearOutput = actf.group(1) == 'linear'
                logging.debug('setting linearOutput to %s' % linearOutput)

    logging.debug('translating lists of weights/thresholds to vectors of pointers')
    weightMatrices = ROOT.vector('TMatrixT<double>*,allocator<TMatrixT<double>*> ')()
    for matrix in weights:
        weightMatrices.push_back(matrix)
    thresholdVectors = ROOT.vector('TVectorT<double>*,allocator<TVectorT<double>*>')()
    for vect in thresholds:
        thresholdVectors.push_back(vect)

    normalizeOutput = True

    logging.debug('building TTrainedNetwork object')
    net = ROOT.TTrainedNetwork(
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

    logging.debug('setting normalization constants')
    offsets = ROOT.vector('double')()
    for u in np.nditer(norm[0]):
        offsets.push_back(-u)
    net.setOffsets(offsets)
    scales = ROOT.vector('double')()
    for s in np.nditer(norm[1]):
        scales.push_back(1.0/s)
    net.setScales(scales)

    return net

def main():
    args = get_args()
    logging.info('Reading network from %s, %s and %s' % (args.model, args.weights, args.normalization))
    model, norm = load_net(args.model, args.weights, args.normalization)
    net = build_ttrained(model, norm)
    logging.info('Writing network to %s' % args.out)
    outf = ROOT.TFile(args.out, 'RECREATE')
    net.Write()
    outf.Close()

if __name__ == '__main__':
    main()
