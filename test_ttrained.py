import numpy as np
import ROOT

import ttrained


def weights(m):
    return [l.get_weights() for l in m.layers if type(l) == 'Dense']

def test(ttrained_path, test_data_path, ntest=1000, error=False):

    tf = ROOT.TFile(ttrained_path)
    tt = tf.Get('TTrainedNetwork')
    mo,no = ttrained.to_keras(tt)
    tt2 = ttrained.from_keras(mo,no)
    mo2,no2 = ttrained.to_keras(tt2)

    assert mo2.to_json() == mo.to_json(), 'model equality'
    assert np.array_equal(no['std'] ,no2['std']) \
        and np.array_equal(no['mean'] ,no2['mean']), 'normalization equality'

    for (w1, b1), (w2, b2) in zip(weights(mo),weights(mo2)):
        assert np.array_equal(w1,w2)
        assert np.array_equal(w2,b2)

    test_file = ROOT.TFile(test_data_path)
    dtree = test_file.Get('NNinput')

    for i in range(10000):
        dtree.GetEntry(i)

        vec = ROOT.vector('Double_t')()
        arr = np.empty(7*7+7+4)

        for j in range(7*7):
            vec.push_back(getattr(dtree, 'NN_matrix{}'.format(j)))
            arr[j] = getattr(dtree, 'NN_matrix{}'.format(j))
        for j in range(7):
            vec.push_back(getattr(dtree, 'NN_pitches{}'.format(j)))
            arr[j+7*7] = getattr(dtree, 'NN_pitches{}'.format(j))
        vec.push_back(dtree.NN_layer)
        arr[-4] = dtree.NN_layer
        vec.push_back(dtree.NN_barrelEC)
        arr[-3] = dtree.NN_barrelEC
        vec.push_back(dtree.NN_phi)
        arr[-2] = dtree.NN_phi
        vec.push_back(dtree.NN_theta)
        arr[-1] = dtree.NN_theta

        if error >= 1:
            arr_ = np.empty(arr.shape[0]+2)
            arr_[0:-2] = arr
            arr_[-2] = dtree.NN_position_id_X_0_pred
            arr_[-1] = dtree.NN_position_id_Y_0_pred
            arr = arr_
            vec.push_back(dtree.NN_position_id_X_0_pred)
            vec.push_back(dtree.NN_position_id_Y_0_pred)
        if error >= 2:
            arr_ = np.empty(arr.shape[0]+2)
            arr_[0:-2] = arr
            arr_[-2] = dtree.NN_position_id_X_1_pred
            arr_[-1] = dtree.NN_position_id_Y_1_pred
            arr = arr_
            vec.push_back(dtree.NN_position_id_X_1_pred)
            vec.push_back(dtree.NN_position_id_Y_1_pred)
        if error >= 3:
            arr_ = np.empty(arr.shape[0]+2)
            arr_[0:-2] = arr
            arr_[-2] = dtree.NN_position_id_X_2_pred
            arr_[-1] = dtree.NN_position_id_Y_2_pred
            arr = arr_
            vec.push_back(dtree.NN_position_id_X_2_pred)
            vec.push_back(dtree.NN_position_id_Y_2_pred)

        arr1 = arr - no['mean']
        arr1 = arr1 / no['std']
        k_out = mo.predict(np.atleast_2d(arr1))
        tt_out = tt.calculateNormalized(vec)
        tt_arr = np.array(tt_out)
        assert np.allclose(tt_arr,k_out)

        tt_out_2 = tt2.calculateNormalized(vec)
        tt_arr_2 = np.array(tt_out)
        assert np.allclose(k_out,tt_arr_2)

        arr2 = arr - no2['mean']
        arr2 = arr2 / no2['std']
        k_out_2 = mo2.predict(np.atleast_2d(arr2))
        assert np.allclose(tt_arr_2, k_out_2)

def main():
    ttrained.init()
    ROOT.gROOT.SetBatch(True)
    print '==> number'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsNumber_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/FIX_NUMBER/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.number.test.root'
    )
    print '==> pos1'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/dev/pixel-NN-training/dump/test/WeightsPosition1_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/train_pos_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.pos1.test.root'
    )
    print '==> pos2'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsPosition2_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/train_pos_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.pos2.test.root'
    )
    print '==> pos3'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsPosition3_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/train_pos_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.pos3.test.root'
    )
    print '==> error1x'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsError_1_x_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/input_error_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.error1.test.root',
        error=1
    )
    print '==> error1y'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsError_1_y_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/input_error_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.error1.test.root',
        error=1
    )
    print '==> error2x'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsError_2_x_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/input_error_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.error2.test.root',
        error=2
    )
    print '==> error2y'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsError_2_y_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/input_error_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.error2.test.root',
        error=2
    )
    print '==> error3x'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsError_3_x_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/input_error_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.error3.test.root',
        error=3
    )
    print '==> error3y'
    test(
        ttrained_path='/lcg/storage15/atlas/gagnon/work/archive/Ref_to_keras/WeightsError_3_y_track.root',
        test_data_path= '/lcg/storage15/atlas/gagnon/work/input_error_v3/user.lgagnon.361026.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6W.AOD_TIDE.e3569_s2608_s2183.ontrack.v3_EXT0.error3.test.root',
        error=3
    )


if __name__ == '__main__':
    main()
