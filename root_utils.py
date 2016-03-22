import thread
import threading

import numpy as np
import ROOT
import root_numpy as rnp


def calc_scale_offset(path, tree, branches):
    tfile = ROOT.TFile(path, 'READ')
    tree = tfile.Get(tree)
    offset = np.zeros(len(branches))
    scales = np.ones(len(branches))
    for i, branch in enumerate(branches):
        arr = rnp.tree2array(tree, branches=branch)
        offset[i] = -np.mean(arr)
        scales[i] = 1.0/np.std(arr)
    return {'scale': scales, 'offset': offset}


# http://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
def root_batch(tree, branches, bounds, scale_offset):
    batch = rnp.tree2array(
        tree=tree,
        branches=branches,
        start=bounds[0],
        stop=bounds[1]
    )
    batch = batch.view(np.float64).reshape(batch.shape + (-1,))
    batch *= scale_offset[0]
    batch += scale_offset[1]
    return batch


# https://github.com/fchollet/keras/issues/1638#issuecomment-179744902
class ThreadsafeIter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, itr):
        self.itr = itr
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.itr.next()


def threadsafe_generator(gen):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def safegen(*a, **kw):
        return ThreadsafeIter(gen(*a, **kw))
    return safegen


@threadsafe_generator
def generator(path,
              tree,
              branches,
              batch=32,
              norm=None,
              train_split=1,
              loop=True): \
              # pylint: disable=too-many-arguments

    tfiles = {}
    trees = {}

    ntrain = int(train_split * get_entries(path, tree))

    if norm is None:
        norm = {'scale': 1, 'offset': 0}

    while True:
        for i in range(0, ntrain, batch):

            thr = thread.get_ident()

            if thr not in trees:
                tfiles[thr] = ROOT.TFile(path, 'READ')
                trees[thr] = tfiles[thr].Get(tree)

            xbatch = root_batch(
                tree=trees[thr],
                branches=branches[0],
                bounds=(i, i+batch),
                scale_offset=(norm['scale'], norm['offset'])
            )

            if len(branches[1]) > 0:
                ybatch = root_batch(
                    tree=trees[thr],
                    branches=branches[1],
                    bounds=(i, i+batch),
                    scale_offset=(1, 0)
                )
            else:
                ybatch = None
            yield (xbatch, ybatch)

        if not loop:
            break


def load_validation(path, tree, branches, norm, validation_split=0):

    tfile = ROOT.TFile(path, 'READ')
    tree = tfile.Get(tree)

    if norm is None:
        norm = {'scale': 1, 'offset': 0}

    nentries = tree.GetEntries()
    start = int(nentries * (1 - validation_split))

    xdat = root_batch(
        tree=tree,
        branches=branches[0],
        bounds=(start, nentries),
        scale_offset=(norm['scale'], norm['offset'])
    )
    ydat = root_batch(
        tree=tree,
        branches=branches[1],
        bounds=(start, nentries),
        scale_offset=(1, 0)
    )

    return xdat, ydat


def get_entries(path, tree):
    tfile = ROOT.TFile(path, 'READ')
    ttree = tfile.Get(tree)
    return ttree.GetEntries()
