CXXFLAGS = -O2 -Wall -Wextra -std=c++11 $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --ldflags --libs)

default: ROC ROC_Graph residuals
all: default TTrainedNetwork.so

residuals: residuals.cxx
	g++ $(CXXFLAGS) -o $@ $< $(ROOTLIBS)

ROC_Graph: ROC_Graph.cxx
	g++ $(CXXFLAGS) -o $@ $< $(ROOTLIBS)

ROC: ROC.cxx
	g++ $(CXXFLAGS) -o $@ $<

TTrainedNetwork.so: TTrainedNetwork.cxx TTrainedNetworkDict.cxx
	g++ $(CXXFLAGS) -shared -fPIC -o $@ $< $(ROOTLIBS)

TTrainedNetworkDict.cxx: Linkdef.h
	rootcint -f $@ -c $<
