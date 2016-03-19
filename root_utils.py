import itertools as it
import thread
import threading

import numpy as np
import ROOT
import root_numpy as rnp

ROOT.gROOT.SetBatch(True)

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

# http://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
def root_batch(tree, branches, start, stop, scale, offset):
    x = rnp.tree2array(tree, branches=branches, start=start, stop=stop)
    x = x.view(np.float64).reshape(x.shape + (-1,))
    x += offset
    x *= scale
    return x

# https://github.com/fchollet/keras/issues/1638#issuecomment-179744902
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generator(path, tree, branches, batch=32, norm=None, train_split=1):

    tfiles = {}
    trees = {}

    ntrain = int(train_split * get_entries(path,tree))

    if norm is None:
        norm = {'scale': 1, 'offset': 0}

    for i in it.cycle(range(0, ntrain, batch)):

        thr = thread.get_ident()

        if not thr in trees:
            tfiles[thr] = ROOT.TFile(path, 'READ')
            trees[thr] = tfiles[thr].Get(tree)

        x = root_batch(trees[thr], branches[0], i, i+batch, norm['scale'], norm['offset'])
        y = root_batch(trees[thr], branches[1], i, i+batch, norm['scale'], norm['offset'])
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

