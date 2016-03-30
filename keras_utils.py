import resource
import sys
import time
import warnings

from keras import activations
from keras.callbacks import Callback, EarlyStopping
from keras.layers.core import MaskedLayer
import numpy as np


class Sigmoid(MaskedLayer):
    def __init__(self, alpha, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.alpha = alpha

    def get_output(self, train=False):
        X = self.get_input(train)
        return activations.sigmoid(self.alpha*X)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'alpha': self.alpha}
        base_config = super(Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Profile(Callback):
    def __init__(self, file=None):
        if file is None:
            self.outfile = sys.stdin
        else:
            self.outfile = open(file, 'w')
        self.time_total = 0
        header = 'epoch,time_epoch,time_total,time_cpu,maxrss\n'
        self.outfile.write(header)

    def on_epoch_begin(self, epoch, logs):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs):
        time_epoch = self.t0 - time.time()
        self.time_total += dt_epoch
        usage = resource.getrusage(resource.RUSAGE_SELF)
        time_cpu = usage.ru_utime + usage.ru_stime
        line = '%d,%f,%f,%f,%d\n' % (epoch,time_epoch,self.time_total,time_cpu,usage.maxrss)
        self.outfile.write(line)


"""This portion of the file implements the variable patience-based
early stopping strategy previously used in the AGILEPack setup. See
e.g.
https://svnweb.cern.ch/cern/wsvn/atlas-rjansky/AGILEPack/trunk/cli-src/train_interface.cxx
around l.250.

The code for the class is an adaptation of the EarlyStopping keras callback
(https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L303, commit 85e51a0)

"""

class ThresholdEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', min_epochs=10,
                 threshold=0.995, increase=1.75, verbose=0, mode='auto'):

        super(ThresholdEarlyStopping, self).__init__(
            monitor=monitor,
            patience=min_epochs,
            verbose=verbose,
            mode=mode
        )

        self.threshold = threshold
        self.increase = increase

    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.patience:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Early stopping requires %s available!' %
                              (self.monitor), RuntimeWarning)

            if self.monitor_op(current, self.best):
                if self.monitor_op(current, self.threshold*self.best):
                    self.patience = max(self.patience, epoch * self.increase)
                self.best = current

        else:
            if self.verbose > 0:
                print('Epoch %05d: early stopping' % (epoch))

            self.model.stop_training = True
