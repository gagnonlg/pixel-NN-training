python2 evalNN_keras.py \
	--input $IN \
	--model $NAME.model.yaml \
	--weights $NAME.weights.hdf5 \
	--config <(python2 genconfig.py --type $TYPE) \
	--output $NAME.db \
	--normalization $NAME.normalization.txt

if [ $TYPE = "pos1" ]; then
   nbins=30
elif [ $TYPE = "pos2" ]; then
   nbins=25
elif [ $TYPE = "pos3" ]; then
   nbins=20
fi

python2 errorNN_input.py \
	--input $IN \
	--db $NAME.db \
	--output $(echo $IN | sed 's/.pos/.error/') \
	--nbins $nbins



