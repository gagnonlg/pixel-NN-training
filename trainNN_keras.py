import argparse
import csv
import os.path
import sys
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np
from ThresholdEarlyStopping import ThresholdEarlyStopping
from pixel_utils import load_data

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--training-input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shape', nargs=2, type=int)
    p.add_argument('--validation-fraction', type=float, default=0.1)
    p.add_argument('--targets', type=int, default=3)
    p.add_argument('--structure', nargs='+', type=int, default=[60,25,20,3])
    p.add_argument('--activation', choices=['sigmoid', 'tanh', 'relu'], default='sigmoid')
    p.add_argument('--output-activation', choices=['softmax', 'linear'], default='softmax')
    p.add_argument('--l2', type=float, default=0.0000001)
    p.add_argument('--learning-rate', type=float, default=0.08)
    p.add_argument('--momentum', type=float, default=0.4)
    p.add_argument('--batch', type=int, default=60)
    p.add_argument('--min-epochs', type=int, default=10)
    p.add_argument('--max-epochs', type=int, default=1000)
    p.add_argument('--patience-increase', type=float, default=1.75)
    p.add_argument('--threshold', type=float, default=0.995)
    p.add_argument('--verbose', default=False, action='store_true')
    return p.parse_args(argv)

def build_model(structure, regularizer, activation, output_activation,
                learning_rate, momentum):
    model = Sequential()
    for i in range(1,len(structure)):

        model.add(Dense(
            input_dim=structure[i-1],
            output_dim=structure[i],
            init='glorot_uniform',
            W_regularizer=l2(regularizer)
            )
        )
        if i < (len(structure) - 1):
            model.add(Activation(activation))
        else:
            model.add(Activation(output_activation))

    loss= 'binary_crossentropy' if output_activation == 'softmax' else 'mse'

    model.compile(
        loss=loss,
        optimizer=SGD(lr=learning_rate, momentum=momentum)
    )

    return model

def early_stopping_callback(min_epochs, threshold, increase,
                            verbose):
     return ThresholdEarlyStopping(
         monitor='val_loss',
         min_epochs=min_epochs,
         threshold=threshold,
         increase=increase,
         mode='min',
         verbose=verbose
     )

def checkpoint_callback(filepath, verbose):
    filepath += '.weights.hdf5'
    return ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=verbose
    )

def normalize_inplace(m,output):

    stats    = np.zeros((2,m.shape[1]))
    stats[0] = m.mean(axis=0)
    stats[1] = m.std(axis=0)

    np.savetxt(stats, output + '.normalization.txt')

    m -= stats[0]
    m /= stats[1]

    return m

def trainNN(training_input,
            validation_fraction,
            output,
            shape=None,
            targets=3,
            structure=[60,25,20,3],
            activation='sigmoid',
            output_activation='softmax',
            l2=0.0000001,
            learning_rate=0.08,
            momentum=0.4,
            batch=60,
            min_epochs=10,
            max_epochs=1000,
            patience_increase=1.75,
            threshold=0.995,
            verbose=False):

    model = build_model(
        structure,
        l2,
        activation,
        output_activation,
        learning_rate,
        momentum
    )

    with open(os.path.splitext(output)[0] + '.model.yaml', 'w') as yfile:
        yfile.write(model.to_yaml())

    callbacks = [
        early_stopping_callback(
            min_epochs,
            threshold,
            patience_increase,
            verbose
        ),
        checkpoint_callback(output, verbose)
    ]

    trainX, trainY, _ = load_data(training_input, targets, shape)

    normalize_inplace(trainX, output)

    model.fit(
        trainX,
        trainY,
        batch_size=batch,
        nb_epoch=max_epochs,
        verbose=(2 if verbose else 0),
        callbacks=callbacks,
        validation_split=validation_fraction,
        shuffle=False
    )

def main(argv):
    args = parse_args(argv)
    trainNN(
        args.training_input,
        args.validation_fraction,
        args.output,
        args.shape,
        args.targets,
        args.structure,
        args.activation,
        args.output_activation,
        args.l2,
        args.learning_rate,
        args.momentum,
        args.batch,
        args.min_epochs,
        args.max_epochs,
        args.patience_increase,
        args.threshold,
        args.verbose
    )


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
