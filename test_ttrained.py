import numpy as np
import ROOT

import ttrained

ROOT.gROOT.SetBatch(True)

ttrained.init()
ref = '/lcg/storage15/atlas/gagnon/dev/pixel-NN-training/dump/test/WeightsPosition1_track.root'
tf = ROOT.TFile(ref)
tt = tf.Get('TTrainedNetwork')
mo,no = ttrained.to_keras(tt)
tt2 = ttrained.from_keras(mo,no)
mo2,no2 = ttrained.to_keras(tt2)

assert mo2.to_json() == mo.to_json(), 'model equality'
assert np.array_equal(no['std'] ,no2['std']) and np.array_equal(no['mean'] ,no2['mean']), 'normalization equality'

def weights(m):
    return [l.get_weights() for l in m.layers if type(l) == 'Dense']

for (w1, b1), (w2, b2) in zip(weights(mo),weights(mo2)):
    assert np.array_equal(w1,w2)
    assert np.array_equal(w2,b2)
