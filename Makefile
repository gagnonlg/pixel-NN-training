CXXFLAGS = -O2 -Wall -Wextra -std=c++11 $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --ldflags --libs)

default: ROC ROC_Graph residuals
ttrained: error_NN_input
all: default ttrained

residuals: residuals.cxx
	g++ $(CXXFLAGS) -o $@ $< $(ROOTLIBS)

ROC_Graph: ROC_Graph.cxx
	g++ $(CXXFLAGS) -o $@ $< $(ROOTLIBS)

ROC: ROC.cxx
	g++ $(CXXFLAGS) -o $@ $<

error_NN_input: TTrainedNetwork.o error_NN_input.cxx
	g++ $(CXXFLAGS) -o $@ $^ $(ROOTLIBS)

TTrainedNetwork.so: TTrainedNetwork.cxx TTrainedNetworkDict.cxx
	g++ $(CXXFLAGS) -shared -fPIC -o $@ $< $(ROOTLIBS)

TTrainedNetwork.o: TTrainedNetwork.cxx TTrainedNetworkDict.cxx
	g++ $(CXXFLAGS) -o $@ -c $< $(ROOTLIBS)

TTrainedNetworkDict.cxx: Linkdef.h
	rootcint -f $@ -c $<
