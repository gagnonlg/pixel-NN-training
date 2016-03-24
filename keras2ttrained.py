import argparse

import keras.models
import ROOT

import ttrained
import utils

ttrained.init()

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--weights', required=True)
p.add_argument('--normalization')
p.add_argument('--output', required=True)
args = p.parse_args()

model = keras.models.model_from_yaml(open(args.model, 'r').read())
model.load_weights(args.weights)
if args.normalization is not None:
    norm = utils.load_normalization(args.normalization)
else:
    norm = None

outf = ROOT.TFile(args.output, 'RECREATE')
net = ttrained.from_keras(model, norm)
net.Write()
outf.Close()
