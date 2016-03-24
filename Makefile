default: ROC ROC_Graph residuals normalization

residuals: residuals.cxx
	g++ -O3 `root-config --cflags` residuals.cxx -o residuals `root-config --libs`

ROC_Graph: ROC_Graph.cxx
	g++ -O3 `root-config --cflags` ROC_Graph.cxx -o ROC_Graph `root-config --libs`

ROC: ROC.cxx
	g++ -O3 ROC.cxx -o ROC
