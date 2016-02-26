import resource
import sys
import time
from keras.callbacks import Callback

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
