"""
This file implements the variable patience-based early stopping
strategy previously used in the AGILEPack setup. See e.g.
https://svnweb.cern.ch/cern/wsvn/atlas-rjansky/AGILEPack/trunk/cli-src/train_interface.cxx
around l.250.

The code for the class is an adaptation of the EarlyStopping keras callback
(https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L303, commit 85e51a0)
"""

import warnings
import numpy as np
from keras.callbacks import EarlyStopping

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
