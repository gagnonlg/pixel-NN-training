from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import keras.regularizers
import numpy as np

from Profile import Profile
from ThresholdEarlyStopping import ThresholdEarlyStopping
import utils
import root_utils


def build_model(structure,
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
            model.add(Activation(activations[0]))
        else:
            model.add(Activation(activations[1]))

    loss = 'binary_crossentropy' if activations[1] == 'softmax' else 'mse'

    model.compile(
        loss=loss,
        optimizer=SGD(lr=learning_rate, momentum=momentum)
    )

    return model


def early_stopping_callback(min_epochs,
                            threshold,
                            increase,
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


def normalize_inplace(data, output, stats=None):

    if stats is None:
        stats = np.zeros((2, data.shape[1]))
        stats[0] = data.mean(axis=0)
        stats[1] = data.std(axis=0)
        np.savetxt(output + '.normalization.txt', stats)

    data -= stats[0]
    data /= stats[1]

    return data


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
             min_epochs=10,
             max_epochs=1000,
             patience_increase=1.75,
             threshold=0.995,
             profile=False,
             nbworkers=1,
             verbose=False): \
            # pylint: disable=too-many-arguments,too-many-locals,dangerous-default-value

    branches = utils.get_data_config_names(config, meta=False)
    norm = root_utils.calc_scale_offset(training_input, 'NNinput', branches[0])
    utils.save_scale_offset(norm, output)
    norm = None

    data_generator = root_utils.generator(
        path=training_input,
        tree='NNinput',
        branches=branches,
        batch=batch,
        norm=norm,
        train_split=(1 - validation_fraction)
    )

    valid_data = root_utils.load_validation(
        path=training_input,
        tree='NNinput',
        branches=branches,
        norm=norm,
        validation_split=validation_fraction
    )

    structure = [len(branches[0])] + structure + [len(branches[1])]

    model = build_model(
        structure,
        regularizer,
        (activation, output_activation),
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
