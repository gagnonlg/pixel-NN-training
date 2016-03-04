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
import utils

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--training-input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--shape', nargs=2, type=int)
    p.add_argument('--validation-fraction', type=float, default=0.1)
    p.add_argument('--structure', nargs='+', type=int, default=[25,20])
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
    p.add_argument('--no-normalize', action='store_true', default=False)
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

    loss= 'binary_crossentropy' if output_activation == 'softmax' else 'mae'

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

    np.savetxt(output + '.normalization.txt', stats)

    m -= stats[0]
    m /= stats[1]

    return m

def trainNN(training_input,
            validation_fraction,
            output,
            config,
            shape=None,
            structure=[25,20],
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
            no_normalize=False,
            verbose=False):

    if shape is None:
        shape = utils.get_shape(training_input, skiprows=1)

    header = utils.get_header(training_input)
    i_inputs, i_targets = utils.get_data_config(config, header, meta=False)
    shape = (shape[0], len(i_inputs) + len(i_targets))
    data = utils.load_data_bulk(training_input, shape, i_select=(i_inputs + i_targets))
    trainX = data[:,:len(i_inputs)]
    trainY = data[:,len(i_inputs):]

    structure = [len(i_inputs)] + structure + [len(i_targets)]

    model = build_model(
        structure,
        l2,
        activation,
        output_activation,
        learning_rate,
        momentum
    )

    with open(output + '.model.yaml', 'w') as yfile:
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

    if not no_normalize:
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
        args.config,
        args.shape,
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
        args.no_normalize,
        args.verbose
    )


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
