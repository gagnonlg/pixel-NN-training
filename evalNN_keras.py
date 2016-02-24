import argparse
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
    p.add_argument("--normalization")
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
    i_inputs -- list with column indices of inputs for the model
    n_targets -- the number of predicted targets.
    """

    X = data[:,i_inputs]
    if normalization is not None:
        X -= normalization[0]
        X /= normalization[1]
    data[:,-ntargets:] = model.predict(X)
    if normalization is not None:
        X *= normalization[1]
        X += normalization[0]


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

def main(argv):
    args = parse_args(argv)
    header = utils.get_header(args.input)
    i_inputs, i_targets, i_meta = utils.get_data_config(args.config, header)
    shape = utils.get_shape(args.input, skiprows=1)
    model = keras.models.model_from_yaml(open(args.model,'r').read())
    model.load_weights(args.weights)
    data = utils.load_data_bulk(args.input, shape, extra=len(i_targets))
    norm = np.loadtxt(args.normalization) if args.normalization else None
    eval_model_inplace(model, data, i_inputs, len(i_targets), norm)
    to_sql(args.output, data, header, i_meta, i_targets)
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))

