""" eval_nn: module to evaluate a dataset using a neural network """

import argparse
import itertools as it
import sqlite3

import keras.models
import numpy as np

import utils
import root_utils

__all__ = ['eval_nn']


def eval_nn(inputp,
            model,
            weights,
            config,
            output,
            normalization): \
            # pylint: disable=too-many-arguments
    """ evaluate a dataset  with a neural network stored on disk

    arguments:
    inputp -- path to the ROOT dataset
    model -- path to the yaml keras model config file
    weights -- path to the hdf5 weights file
    config -- path to the branches config file
    output -- output name for the sqlite database (overwrites the 'test' table)
    normalization -- path to the txt file with normalization constants
    """
    model = keras.models.model_from_yaml(open(model, 'r').read())
    model.load_weights(weights)

    _eval_dataset(
        model=model,
        path=inputp,
        tree='NNinput',
        branches=utils.get_data_config_names(config, meta=True),
        norm=utils.load_normalization(normalization),
        dbpath=output
    )


def _eval_dataset(model,
                  path,
                  tree,
                  branches,
                  norm,
                  dbpath,
                  batch=128): \
                  # pylint: disable=too-many-arguments

    dbconn = sqlite3.connect(dbpath)
    _prepare_db(
        dbconn=dbconn,
        meta_branches=branches[2],
        y_branches=branches[1]
    )

    data_generator = root_utils.generator(
        path,
        tree=tree,
        branches=branches[:2],
        batch=batch,
        normalization=norm,
        loop=False
    )

    meta_generator = root_utils.generator(
        path,
        tree=tree,
        branches=(branches[2], []),
        batch=batch,
        normalization=None,
        loop=False
    )

    for (xbatch, ybatch), (meta, _) in it.izip(data_generator, meta_generator):
        _insert_into_db(
            dbconn,
            meta,
            y_truth=ybatch,
            y_pred=model.predict(xbatch, batch_size=xbatch.shape[0])
        )

    dbconn.commit()


def _insert_into_db(dbconn, meta, y_truth, y_pred):

    mcol = meta.shape[1]
    ycol = y_truth.shape[1]

    shape = (meta.shape[0], mcol + 2*ycol)

    meta_begin = 0
    meta_end = meta_begin + mcol
    truth_begin = meta_end
    truth_end = truth_begin + ycol
    pred_begin = truth_end
    pred_end = pred_begin + ycol

    data = np.empty(shape)
    data[:, meta_begin:meta_end] = meta
    data[:, truth_begin:truth_end] = y_truth
    data[:, pred_begin:pred_end] = y_pred

    sql = 'INSERT INTO test VALUES ('
    sql += ','.join(['?'] * shape[1])
    sql += ')'

    dbconn.executemany(sql, data)


def _prepare_db(dbconn, meta_branches, y_branches):

    dbconn.execute('DROP TABLE IF EXISTS test')
    sql = 'CREATE TABLE test ('
    sql += ','.join([br + ' REAL' for br in meta_branches])
    sql += ','
    sql += ','.join([br + '_TRUTH REAL' for br in y_branches])
    sql += ','
    sql += ','.join([br + '_PRED REAL' for br in y_branches])
    sql += ')'
    dbconn.execute(sql)
    dbconn.commit()


def _main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", required=True)
    parse.add_argument("--model", required=True)
    parse.add_argument("--weights", required=True)
    parse.add_argument("--config", required=True)
    parse.add_argument("--output", required=True)
    parse.add_argument("--normalization", required=True)
    args = parse.parse_args()

    eval_nn(
        args.input,
        args.model,
        args.weights,
        args.config,
        args.output,
        args.normalization
    )


if __name__ == '__main__':
    _main()
