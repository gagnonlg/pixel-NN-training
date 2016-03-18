import numpy as np
import ROOT
import root_numpy as rnp

def scale_offset(path, tree, branches):
    tfile = ROOT.TFile(path, 'READ')
    tree = tfile.Get(tree)
    offset = np.zeros(len(branches))
    scales = np.ones(len(branches))
    for i,b in enumerate(branches):
        r = rnp.tree2array(tree,branches=b)
        offset[i] = -np.mean(r)
        scales[i] = -np.std(r)
    return {'scale': scales, 'offset': offset}

def root_batch(tree, branches, start, stop, scale, offset):
    x = rnp.tree2array(tree, branches=branches, start=start, stop=stop)
    x = x.view(np.float64).reshape(x.shape + (-1,))
    x += offset
    x *= scale
    return x

def generator(path, tree, branches, batch=32, norm=None, train_split=1):

    tfile = ROOT.TFile(path, 'READ')
    tree = tfile.Get(tree)

    ntrain = int(train_split * tree.GetEntries())

    if norm is None:
        norm = {'scale': 1, 'offset': 0}

    while True:
        for i in range(0, ntrain, batch):
            x = root_batch(tree, branches[0], i, i+batch, norm['scale'], norm['offset'])
            y = root_batch(tree, branches[1], i, i+batch, norm['scale'], norm['offset'])
            yield (x,y)

# TODO change this to load_data
def load_validation(path, tree, branches, norm, validation_split=0):

    tfile = ROOT.TFile(path, 'READ')
    tree = tfile.Get(tree)

    if norm is None:
        norm = {'scale': 1, 'offset': 0}

    nentries = tree.GetEntries()
    start = int(nentries * (1 - validation_split))

    x = root_batch(tree, branches[0], start, nentries, norm['scale'], norm['offset'])
    y = root_batch(tree, branches[1], start, nentries, norm['scale'], norm['offset'])

    return x,y

def get_entries(path,tree):
    tf = ROOT.TFile(path,'READ')
    tr = tf.Get(tree)
    return tr.GetEntries()

