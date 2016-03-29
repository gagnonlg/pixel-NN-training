# Script for batch submission
# e.g.
# qsub -v "NAME=test,TRAINING=train.root,TEST=test.root,TYPE=pos1,SIZEX=7,SIZEY=7" -N test -d $PWD -j oe launch.sh

. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
. $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh root

export PATH=./pixel-NN-training:$PATH
export NJOBS=1

set -u
set -e

if [ $TYPE = "number" ]
then
    python2 ./pixel-NN-training/trainNN_keras.py \
	--training-input $TRAINING \
	--output $NAME \
        --config <(python2 pixel-NN-training/genconfig.py --type $TYPE --sizeX $SIZEX --sizeY $SIZEY) \
        --structure 25 20 \
        --output-activation softmax \
	--l2 0.0000001 \
        --learning-rate 0.08 \
	--momentum 0.4 \
	--batch 60 \
	--verbose
else
    python2 ./pixel-NN-training/evalNN_keras.py \
	--training-input $TRAINING \
	--output $NAME \
	--config <(python2 pixel-NN-training/genconfig.py --type $TYPE --sizeX $SIZEX --sizeY $SIZEY) \
        --structure 40 20 \
        --output-activation linear \
	--l2 0.0000001 \
        --learning-rate 0.04 \
	--momentum 0.3 \
	--batch 30 \
	--verbose
fi

evalNN --input $TEST \
       --model $NAME.model.yaml \
       --weights $NAME.weights.hdf5 \
       --config <(python2 pixel-NN-training/genconfig.py --type $TYPE --sizeX $SIZEX --sizeY $SIZEY) \
       --output $NAME.db \
       --normalization $NAME.normalization.txt

test-driver $TYPE $NAME.db $NAME.root $SIZEY 
