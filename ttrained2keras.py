import argparse

import ROOT

import ttrained
import utils

ttrained.init()

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--sigmoid2', action='store_true')
args = p.parse_args()

tfile = ROOT.TFile(args.input, 'READ')
net = tfile.Get('TTrainedNetwork')
model, norm = ttrained.to_keras(net, sigmoid2=args.sigmoid2)

with open(args.output + '.model.yaml', 'w') as yfile:
    yfile.write(model.to_yaml())

model.save_weights(args.output + '.weights.hdf5', overwrite=True)
utils.save_normalization(norm, args.output + '.normalization.txt')
