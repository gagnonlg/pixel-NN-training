""" train_nn: module to train a neural network """

import argparse

from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import keras.regularizers

from keras_utils import Sigmoid, Profile, ThresholdEarlyStopping
import root_utils
import utils


__all__ = ['train_nn']


def train_nn(training_input,
             validation_fraction,
             output,
             config,
             structure=[25, 20],
             activation='sigmoid',
             output_activation='softmax',
             regularizer=0.0000001,
             learning_rate=0.08,
             momentum=0.4,
             batch=60,
             max_epochs=1000,
             patience=5,
             profile=False,
             nbworkers=1,
             save_all=False,
             verbose=False): \
            # pylint: disable=too-many-arguments,too-many-locals,dangerous-default-value
    """ train a neural network

    arguments:
    training_input -- path to ROOT training dataset
    validation_fraction -- held-out fraction of dataset for early-stopping
    output -- prefix for output files
    structure -- list with number of hidden nodes in hidden layers
    activation -- non-linearity for hidden layers
    output_activation -- non-linearity of output of the network
    regularizer -- l2 weight regularizer
    momentum -- momentum of weight updates
    batch -- minibatch size
    max_epochs -- maximum number of training epochs
    patience -- early stopping tolerance, in number of epochs
    profile -- create a memory usage log
    nbworkers -- number of parallel thread to load minibatches
    save_all -- save weights at each epoch
    verbose -- bla bla bla level
    """
    branches = utils.get_data_config_names(config, meta=False)
    norm = root_utils.calc_normalization(
        training_input,
        'NNinput',
        branches[0]
    )
    utils.save_normalization(norm, output)

    data_generator = root_utils.generator(
        path=training_input,
        tree='NNinput',
        branches=branches,
        batch=batch,
        normalization=norm,
        train_split=(1 - validation_fraction)
    )

    valid_data = root_utils.load_validation(
        path=training_input,
        tree='NNinput',
        branches=branches,
        normalization=norm,
        validation_split=validation_fraction
    )

    structure = [len(branches[0])] + structure + [len(branches[1])]

    model = _build_model(
        structure,
        regularizer,
        (activation, output_activation),
        learning_rate,
        momentum
    )

    with open(output + '.model.yaml', 'w') as yfile:
        yfile.write(model.to_yaml())

    callbacks = [
        _early_stopping_callback(
            patience,
            verbose
        ),
        _checkpoint_callback(output, verbose)
    ]

    if profile:
        callbacks.append(Profile('%s.profile.txt' % output))

    if save_all:
        name = output
        if name.endswith('hdf5'):
            name = name.replace('hdf5', '{epoch:03d}.hdf5')
        else:
            name += '.{epoch:03d}.hdf5'
        callbacks.append(ModelCheckpoint(name))

    nentries = root_utils.get_entries(training_input, 'NNinput')

    model.fit_generator(
        generator=data_generator,
        samples_per_epoch=int(nentries * (1 - validation_fraction)),
        nb_epoch=max_epochs,
        verbose=(2 if verbose else 0),
        callbacks=callbacks,
        validation_data=valid_data,
        nb_worker=nbworkers
    )


def _build_model(structure,
                 regularizer,
                 activations,
                 learning_rate,
                 momentum):
    model = Sequential()
    for i in range(1, len(structure)):

        model.add(
            Dense(
                input_dim=structure[i-1],
                output_dim=structure[i],
                init='glorot_uniform',
                W_regularizer=keras.regularizers.l2(regularizer)
            )
        )
        if i < (len(structure) - 1):
            if activations[0] == 'sigmoid2':
                model.add(Sigmoid(2))
            else:
                model.add(Activation(activations[0]))
        elif activations[1] == 'sigmoid2':
            model.add(Sigmoid(2))
        else:
            model.add(Activation(activations[1]))

    loss = 'categorical_crossentropy' if activations[1] != 'linear' else 'mae'

    model.compile(
        loss=loss,
        optimizer=SGD(lr=learning_rate, momentum=momentum)
    )

    return model


def _early_stopping_callback(patience, verbose):
    return keras.callbacks.EarlyStopping(
        patience=patience,
        verbose=verbose
    )


def _checkpoint_callback(filepath, verbose):
    filepath += '.weights.hdf5'
    return ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=verbose
    )


def _main():

    parse = argparse.ArgumentParser()
    parse.add_argument('--training-input', required=True)
    parse.add_argument('--output', required=True)
    parse.add_argument('--config', required=True)
    parse.add_argument('--validation-fraction', type=float, default=0.1)
    parse.add_argument('--structure', nargs='+', type=int, default=[25, 20])
    parse.add_argument('--activation', choices=['sigmoid', 'sigmoid2', 'tanh', 'relu'], default='sigmoid')
    parse.add_argument('--output-activation', choices=['softmax', 'linear', 'sigmoid', 'sigmoid2'], default='softmax')
    parse.add_argument('--l2', type=float, default=0.0000001)
    parse.add_argument('--learning-rate', type=float, default=0.08)
    parse.add_argument('--momentum', type=float, default=0.4)
    parse.add_argument('--batch', type=int, default=60)
    parse.add_argument('--max-epochs', type=int, default=1000)
    parse.add_argument('--patience', type=int, default=5)
    parse.add_argument('--profile', default=False, action='store_true')
    parse.add_argument('--nbworkers', default=1, type=int)
    parse.add_argument('--save-all', action='store_true', default=False)
    parse.add_argument('--verbose', default=False, action='store_true')
    args = parse.parse_args()

    train_nn(
        args.training_input,
        args.validation_fraction,
        args.output,
        args.config,
        args.structure,
        args.activation,
        args.output_activation,
        args.l2,
        args.learning_rate,
        args.momentum,
        args.batch,
        args.max_epochs,
        args.patience,
        args.profile,
        args.nbworkers,
        args.save_all,
        args.verbose
    )


if __name__ == '__main__':
    _main()
