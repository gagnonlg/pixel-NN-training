default: ROC ROC_Graph residuals normalization

normalization: normalization.cxx
	g++ -O3 -std=c++11 $< -o $@ -L/home/zp/gagnon/local/lib -lgsl -lgslcblas

residuals: residuals.cxx
	g++ -O3 `root-config --cflags` residuals.cxx -o residuals `root-config --libs`

ROC_Graph: ROC_Graph.cxx
	g++ -O3 `root-config --cflags` ROC_Graph.cxx -o ROC_Graph `root-config --libs`

ROC: ROC.cxx
	g++ -O3 ROC.cxx -o ROC
