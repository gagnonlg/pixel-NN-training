from keras.layers.core import Dense, Activation
from keras.models import Sequential
import numpy as np
import ROOT

ROOT.gROOT.ProcessLine('.L TTrainedNetwork.so')

tfile = ROOT.TFile('ref/WeightsPosition1_track.root')
net = tfile.Get('TTrainedNetwork')

weights = []
for matrix in net.weightMatrices():
    weights.append(np.zeros((matrix.GetNrows(),matrix.GetNcols())))
    for i in range(matrix.GetNrows()):
        for j in range(matrix.GetNcols()):
            weights[-1][i,j] = matrix(i,j)

thresholds = []
for vect in net.getThresholdVectors():
    thresholds.append(np.zeros(vect.GetNrows()))
    for i in range(vect.GetNrows()):
        thresholds[-1][i] = vect(i)

struct = [net.getnInput()] + list(net.getnHiddenLayerSize()) + [net.getnOutput()]

model = Sequential()


for i in range(1, len(struct)):
    model.add(Dense(input_dim=struct[i-1], output_dim=struct[i]))
    model.layers[-1].set_weights([weights[i-1], thresholds[i-1]])
    if i < (len(struct) - 1):
        model.add(Activation('sigmoid'))
    else:
        act = 'linear' if net.getIfLinearOutput() else 'softmax'
        model.add(Activation(act))

