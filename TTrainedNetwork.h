#ifndef TTRAINEDNETWORK__HH
#define TTRAINEDNETWORK__HH

#include "TObject.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include <math.h>
#include <vector>
#include <string>
#include <map>


/******************************************************
  @class TTrainedNetwork
  Created : 18-02-2008
  @author Giacinto Piacquadio (giacinto.piacquadio AT physik.uni-freiburg.de)
********************************************************/

namespace nnopt {
}

class TTrainedNetwork : public TObject
{
public:

  static const unsigned linearOutput = 1u << 0;
  static const unsigned normalizeOutput = 1u << 1;

  enum ActivationFunction {
    SIGMOID = 1
  };

  struct Input {
    std::string name; //<- requires unique strings or none at all
    double offset; //<- this value is added to the input before giving to nn
    double scale; //<- after offset is added, input is scaled by this value
  };

  typedef std::vector<Double_t> DVec;
  typedef std::map<std::string, double> DMap;
  typedef DMap::const_iterator DMapI;

  TTrainedNetwork();

  //NOTE: class takes ownership of all pointers.

  // old-school constructor (for compatability)
  TTrainedNetwork(Int_t nInput,
                 Int_t nHidden,
                 Int_t nOutput,
                 std::vector<Int_t> & nHiddenLayerSize,
                 std::vector<TVectorD*> & thresholdVectors,
                 std::vector<TMatrixD*> & weightMatrices,
                 Int_t activationFunction,
                 bool linearOutput = false,
                 bool normalizeOutput = false);

  // You can also add offsets and scales.
  // This is intended as a workaround for older code that didn't include
  // normalization.
  void setOffsets(const std::vector<double>& offsets);
  void setScales(const std::vector<double>& scales);

  // new constructor for normalized nn.
  // This avoids some chances for logical inconsistency by constructing
  // the hidden layer size from the thresholdVectors and weightMatrices.
  // Also runs a consistency check on thresholdVectors and weightMatrices.
  TTrainedNetwork(std::vector<TTrainedNetwork::Input> inputs,
                 unsigned nOutput,
                 std::vector<TVectorD*> & thresholdVectors,
                 std::vector<TMatrixD*> & weightMatrices,
                 int activationFunction = SIGMOID,
                 unsigned options = 0);

  ~TTrainedNetwork();

  // returns an empty vector if normalization isn't set
  std::vector<Input> getInputs() const;

  void setNewWeights(std::vector<TVectorD*> & thresholdVectors,
                     std::vector<TMatrixD*> & weightMatrices);

  Int_t getnInput() const {return mnInput;};

  Int_t getnHidden() const {return mnHidden;};

  Int_t getnOutput() const {return mnOutput;};

  const std::vector<Int_t> &  getnHiddenLayerSize() const {
    return mnHiddenLayerSize;
  };

  Int_t getActivationFunction() const {return mActivationFunction;};

  const std::vector<TVectorD*> & getThresholdVectors() const {
    return mThresholdVectors;
  };

  const std::vector<TMatrixD*> & weightMatrices() const {
    return mWeightMatrices;
  };

  // these methods should be optimized.
  DVec calculateOutputValues(const DVec & input) const;
  DVec calculateNormalized(const DVec& input) const;

  // these are intentionally slow: the NN must
  // rearrange the inputs each time these functions are called.
  // They are designed for robustness and ease of use, not for highly
  // optimized code.
  DVec calculateNormalized(const DMap & input) const;

  bool getIfLinearOutput() const { return mLinearOutput; };

  bool getIfNormalizeOutput() const { return mNormalizeOutput; };

private:


  unsigned mnInput;
  unsigned mnHidden;
  unsigned mnOutput;

  // in an ideal world these would be one object in a vector, but
  // storing classes within classes in root is ugly
  std::vector<Double_t> m_input_node_offset;
  std::vector<Double_t> m_input_node_scale;

  std::map<std::string,int> m_inputStringToNode;

  std::vector<Int_t> mnHiddenLayerSize;

  std::vector<TVectorD*> mThresholdVectors;
  std::vector<TMatrixD*> mWeightMatrices;

  Int_t mActivationFunction;

  bool mLinearOutput;

  bool mNormalizeOutput;

  double maxExpValue;

  Double_t sigmoid(Double_t x) const;

  // assertions to check for nonsense initialization
  bool is_consistent() const;
  bool check_norm_size(unsigned size) const;

  ClassDef(TTrainedNetwork, 3 )

};

#endif
