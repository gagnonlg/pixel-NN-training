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
from Profile import Profile
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
    p.add_argument('--use-generator', default=False, action='store_true')
    p.add_argument('--normalization')
    p.add_argument('--profile', default=False, action='store_true')
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

def normalize_inplace(m,output,stats=None):

    if stats is None:
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
            use_generator=False,
            normalization=None,
            profile=False,
            verbose=False):

    if shape is None:
        shape = utils.get_shape(training_input, skiprows=1)
    header = utils.get_header(training_input)
    i_inputs, i_targets = utils.get_data_config(config, header, meta=False)

    if use_generator:

        if normalization is not None:
            norm = np.loadtxt(normalization)
        else:
            norm = utils.calc_normalization(training_input)

        nvalid = int(validation_fraction * shape[0])

        data_generator = utils.load_generator(
            path=training_input,
            skiprows=nvalid,
            batch=batch,
            i_inputs=i_inputs,
            i_targets=i_targets,
            norm=norm
        )

        valid_data = utils.load_data_bulk(training_input, (nvalid,shape[1]))
        vX = valid_data[:,i_inputs]
        vY = valid_data[:,i_targets]
        valid_data = (vX,vY)

    else:
        data = utils.load_data_bulk(training_input, shape)
        trainX = data[:,i_inputs]
        trainY = data[:,i_targets]

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

    if profile:
        callbacks.append(Profile('%s.profile.txt' % output))

    if (not use_generator) and (not no_normalize):
        if normalize is not None:
            norm = np.loadtxt(normalization)
        else:
            norm=None
        normalize_inplace(trainX, output, norm[:,i_inputs])

    if use_generator:
        model.fit_generator(
            generator=data_generator,
            samples_per_epoch=int(shape[0] * (1 - validation_fraction)),
            nb_epoch=max_epochs,
            verbose=(2 if verbose else 0),
            callbacks=callbacks,
            validation_data=valid_data
        )
    else:
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
        args.use_generator,
        args.normalization,
        args.profile,
        args.verbose
    )


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
