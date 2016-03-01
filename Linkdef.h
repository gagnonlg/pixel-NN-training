#ifdef __CINT__

#include "TTrainedNetwork.h"
#include <vector>
#include <TMatrixT.h>
#include <TVectorT.h>

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class vector<TMatrixT<double>>;
#pragma link C++ class vector<TMatrixT<double>*,allocator<TMatrixT<double>*> >;
#pragma link C++ class TTrainedNetwork;
#pragma link C++ class vector<TVectorT<double>*,allocator<TVectorT<double>*> >;
#pragma link C++ class vector<TTrainedNetwork::Input,allocator<TTrainedNetwork::Input> >;
#pragma link C++ class TTrainedNetwork::Input;

#endif
