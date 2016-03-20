import argparse
import itertools as it
import root_utils
import sqlite3
import subprocess
import sys
import tempfile
import keras.models
import numpy as np
import utils


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale-offset", required=True)
    return p.parse_args(argv)


def eval_model_inplace(model, data, i_inputs, ntargets, normalization=None):
    """
    evaluate a dataset with the given model and store the results in-place in data array

    model -- the keras model to be used for evaluation
    data  -- the numpy array containing the dataset. It is assumed that the
             last <n_targets> columns are place-holders that are to be
             filled with the predictions
    normalization -- numpy array (shape=(2,len(<i_inputs)) with means in
                    first row and standard deviations in second.
                    **not** performed in-place.
    i_inputs -- list with column indices of inputs for the model
    n_targets -- the number of predicted targets.
    """

    X = data[:,i_inputs]
    if normalization is not None:
        X = (X - normalization[0]) / normalization[1]
    data[:,-ntargets:] = model.predict(X)

def to_sql(fname, data, colnames, i_metadata, i_targets):

    tbl = "CREATE TABLE test ("
    for c in [h for (i,h) in enumerate(colnames) if  i in i_metadata]:
        tbl += "%s REAL, " % c
    for c in [h for (i,h) in enumerate(colnames) if  i in i_targets]:
        tbl += "%s_TRUTH REAL, " % c
    for c in [h for (i,h) in enumerate(colnames) if  i in i_targets]:
        tbl += "%s_PRED REAL," % c
    tbl = tbl[:-1] + ");"

    i_pred = range(data.shape[1] - len(i_targets), data.shape[1])
    with tempfile.NamedTemporaryFile() as tmp:
        np.savetxt(tmp.name,data[:, i_metadata + i_targets + i_pred])
        proc = subprocess.Popen(['sqlite3', fname], stdin=subprocess.PIPE)
        proc.communicate(tbl + '\n' + '.separator " "\n.import %s test\n' % tmp.name)

def insert_into_db(db, meta, y_truth, y_pred):

    mcol = meta.shape[1]
    ycol = y_truth.shape[1]

    nrow = meta.shape[0]
    ncol = mcol + 2*ycol

    meta_begin = 0
    meta_end = meta_begin + mcol
    truth_begin = meta_end
    truth_end = truth_begin + ycol
    pred_begin = truth_end
    pred_end = pred_begin + ycol

    data = np.empty((nrow, ncol))
    data[:,meta_begin:meta_end] = meta
    data[:,truth_begin:truth_end] = y_truth
    data[:,pred_begin:pred_end] = y_pred

    sql = 'INSERT INTO test VALUES ('
    sql += ','.join(['?'] * ncol)
    sql += ')'

    c = db.cursor()
    c.executemany(sql, data)

def prepare_db(db, meta_branches, y_branches):
    cur = db.cursor()
    cur.execute('DROP TABLE IF EXISTS test')
    sql  = 'CREATE TABLE test ('
    sql += ','.join(map(lambda s: s + ' REAL', meta_branches))
    sql += ','
    sql += ','.join(map(lambda s: s + '_TRUTH REAL', y_branches))
    sql += ','
    sql += ','.join(map(lambda s: s + '_PRED REAL', y_branches))
    sql += ')'
    cur.execute(sql)
    db.commit()

def eval_dataset(model,
                 path,
                 tree,
                 branches,
                 meta_branches,
                 norm,
                 dbpath,
                 batch=128):

    db = sqlite3.connect(dbpath)
    prepare_db(
        db=db,
        meta_branches=meta_branches,
        y_branches=branches[1]
    )

    data_generator = root_utils.generator(
        path,
        tree=tree,
        branches=branches,
        batch=batch,
        norm=norm,
        loop=False
    )

    meta_generator = root_utils.generator(
        path,
        tree=tree,
        branches=(meta_branches,[]),
        batch=batch,
        norm=None,
        loop=False
    )

    for (x, y), (meta,_) in it.izip(data_generator,meta_generator):
        ypred = model.predict(x, batch_size=x.shape[0])
        insert_into_db(db, meta, y, ypred)

    db.commit()

def main(argv):
    args = parse_args(argv)
    inputs, targets, meta = utils.get_data_config_names(args.config, meta=True)
    model = keras.models.model_from_yaml(open(args.model,'r').read())
    model.load_weights(args.weights)

    eval_dataset(
        model=model,
        path=args.input,
        tree='NNinput',
        branches=(inputs,targets),
        meta_branches=meta,
        norm=utils.load_scale_offset(args.scale_offset),
        dbpath=args.output
    )

    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))

