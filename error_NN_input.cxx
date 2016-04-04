#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>

#include <TTrainedNetwork.h>


struct Input {
	double sizeX;
	double sizeY;
	double layer;
	double barrelEC;
	double phi;
	double theta;
	double localEtaPixelIndexWeightedPosition;
	double localPhiPixelIndexWeightedPosition;
	std::vector<double> matrix;
	std::vector<double> pitches;
	std::vector<double> positions_id_X;
	std::vector<double> positions_id_Y;
};

void connect(Input *inp, TTree *tree, int nparticles)
{
	tree->SetBranchStatus("*", 0);

	tree->SetBranchStatus("NN_layer", 1);
	tree->SetBranchAddress("NN_layer", &(inp->layer));

	tree->SetBranchStatus("NN_barrelEC", 1);
	tree->SetBranchAddress("NN_barrelEC", &(inp->barrelEC));

	tree->SetBranchStatus("NN_phi", 1);
	tree->SetBranchAddress("NN_phi", &(inp->phi));

	tree->SetBranchStatus("NN_theta", 1);
	tree->SetBranchAddress("NN_theta", &(inp->theta));

	tree->SetBranchStatus("NN_localEtaPixelIndexWeightedPosition", 1);
	tree->SetBranchAddress("NN_localEtaPixelIndexWeightedPosition", &(inp->localEtaPixelIndexWeightedPosition));

	tree->SetBranchStatus("NN_localPhiPixelIndexWeightedPosition", 1);
	tree->SetBranchAddress("NN_localPhiPixelIndexWeightedPosition", &(inp->localPhiPixelIndexWeightedPosition));
	tree->SetBranchStatus("NN_sizeX", 1);
	tree->SetBranchAddress("NN_sizeX", &(inp->sizeX));
	tree->SetBranchStatus("NN_sizeY", 1);
	tree->SetBranchAddress("NN_sizeY", &(inp->sizeY));

	tree->GetEntry(0);

	inp->matrix.resize(inp->sizeX * inp->sizeY);
	for (int i = 0; i < (inp->sizeX * inp->sizeY); i++) {
		char key[256];
		std::sprintf(key, "NN_matrix%d", i);
		tree->SetBranchStatus(key, 1);
		tree->SetBranchAddress(key, &(inp->matrix.at(i)));
	}

	inp->pitches.resize(inp->sizeY);
	for (int i = 0; i < inp->sizeY; i++) {
		char key[256];
		std::sprintf(key, "NN_pitches%d", i);
		tree->SetBranchStatus(key, 1);
		tree->SetBranchAddress(key, &(inp->pitches.at(i)));
	}

	inp->positions_id_X.resize(nparticles);
	inp->positions_id_Y.resize(nparticles);

	if (nparticles >= 1) {
		tree->SetBranchStatus("NN_position_id_X_0", 1);
		tree->SetBranchAddress("NN_position_id_X_0", &(inp->positions_id_X.at(0)));
		tree->SetBranchStatus("NN_position_id_Y_0", 1);
		tree->SetBranchAddress("NN_position_id_Y_0", &(inp->positions_id_Y.at(0)));
	}
	if (nparticles >= 2) {
		tree->SetBranchStatus("NN_position_id_X_1", 1);
		tree->SetBranchAddress("NN_position_id_X_1", &(inp->positions_id_X.at(1)));
		tree->SetBranchStatus("NN_position_id_Y_1", 1);
		tree->SetBranchAddress("NN_position_id_Y_1", &(inp->positions_id_Y.at(1)));
	}
	if (nparticles >= 3) {
		tree->SetBranchStatus("NN_position_id_X_2", 1);
		tree->SetBranchAddress("NN_position_id_X_2", &(inp->positions_id_X.at(2)));
		tree->SetBranchStatus("NN_position_id_Y_2", 1);
		tree->SetBranchAddress("NN_position_id_Y_2", &(inp->positions_id_Y.at(2)));
	}
}


double correctedX(double center_pos,
		  double pos_pixels)
{
  double pitch = 0.05;
  return center_pos + pos_pixels * pitch;
}


double correctedY(double center_pos,
		  double pos_pixels,
		  double size_Y,
		  std::vector<double>& pitches)
{
  double p = pos_pixels + (size_Y - 1) / 2.0;
  double p_Y = -100;
  double p_center = -100;
  double p_actual = 0;

  for (int i = 0; i < size_Y; i++) {
    if (p >= i && p <= (i + 1))
      p_Y = p_actual + (p - i + 0.5) * pitches.at(i);
    if (i == (size_Y - 1) / 2)
      p_center = p_actual + 0.5 * pitches.at(i);
    p_actual += pitches.at(i);
  }

  return center_pos + p_Y - p_center;
}


void fill_amplitude(double res, std::vector<double> *vect, double max)
{
	double sf = 1.0 / (2.0 * max);
	double epsilon = std::numeric_limits<double>::epsilon();
	int  amp = (res + max) * sf * (vect->size() - 1) - epsilon;
	if (amp < 0)
		amp = 0;
	if ((size_t)amp > (vect->size() - 1))
		amp = vect->size() - 1;

	for (size_t i = 0; i < vect->size(); i++) {
		vect->at(i) = (double)(i == (size_t)amp);
	}
}


void prepare_output(Input &inp, std::vector<double> &outp, int nparticles,
		    std::vector<std::vector<double>*> *residualsX,
		    std::vector<std::vector<double>*> *residualsY)
{

	double centerposX = inp.localEtaPixelIndexWeightedPosition;
	double centerposY = inp.localPhiPixelIndexWeightedPosition;
	double x_pred, y_pred, x_truth, y_truth;

	for (int i = 0; i < nparticles; i++) {
		x_pred = outp.at(i*2);
		y_pred = outp.at(i*2 + 1);
		x_truth = inp.positions_id_X.at(i);
		y_truth = inp.positions_id_Y.at(i);

		x_pred = correctedX(centerposX, x_pred);
		x_truth = correctedX(centerposX, x_truth);

		y_pred = correctedY(centerposY,
				    y_pred,
				    inp.sizeY,
				    inp.pitches);
		y_truth = correctedY(centerposY,
				     y_truth,
				     inp.sizeY,
				     inp.pitches);

		/* X */
		double max = (i == 0)? 0.03 : 0.05;
		fill_amplitude(x_pred - x_truth, residualsX->at(i), max);
		/* Y */
		max = (i == 0)? 0.3 : 0.5;
		fill_amplitude(x_pred - x_truth, residualsY->at(i), max);
	}
}


std::vector<double> get_TTrained_input(Input &inp)
{
	std::vector<double> inputData;

	for (int j = 0; j < (int)inp.matrix.size(); j++)
		inputData.push_back(inp.matrix.at(j));
	for (int j = 0; j < (int)inp.pitches.size(); j++)
		inputData.push_back(inp.pitches.at(j));

	inputData.push_back(inp.layer);
	inputData.push_back(inp.barrelEC);
	inputData.push_back(inp.phi);
	inputData.push_back(inp.theta);

	return inputData;
}


int main(int argc, char *argv[])
{
	if (argc != 6) {
		std::fprintf(stderr, "usage: %s <inrootfile> <NNrootfile> <outrootfile> <nparticles> <nbins>", argv[0]);
		return 1;
	}

	TFile in(argv[1], "READ");
	TTree *orig = (TTree*)in.Get("NNinput");

	TFile net_in(argv[2], "READ");
	TTrainedNetwork *net = (TTrainedNetwork*)net_in.Get("TTrainedNetwork");

	TFile out(argv[3], "RECREATE");

	TTree *tree = (TTree*)orig->CloneTree();
	Input inp;
	int nparticles = std::atoi(argv[4]);
	int nbins = std::atoi(argv[5]);

	connect(&inp, tree, nparticles);

	std::vector<TBranch*> bpts_x;
	std::vector<TBranch*> bpts_y;

	std::vector<std::vector<double>*> *residualsX = new std::vector<std::vector<double>*>;
	std::vector<std::vector<double>*> *residualsY = new std::vector<std::vector<double>*>;
	for (int i = 0; i < nparticles; i++) {
		residualsX->push_back(new std::vector<double>(nbins));
		residualsY->push_back(new std::vector<double>(nbins));
		for (int j = 0; j < nbins; j++) {
			char key[256];
			std::sprintf(key, "NN_error_X_%d_%d", i, j);
			bpts_x.push_back(tree->Branch(key, &(residualsX->at(i)->at(j))));
			std::sprintf(key, "NN_error_Y_%d_%d", i, j);
			bpts_y.push_back(tree->Branch(key, &(residualsY->at(i)->at(j))));
		}
	}

	std::vector<double> pred(nparticles * 2);
	for (int i = 0; i < nparticles; i++) {
		char key[256];
		std::sprintf(key, "NN_position_id_X_%d_pred", i*2);
		bpts_x.push_back(tree->Branch(key, &pred.at(i*2)));
		std::sprintf(key, "NN_position_id_Y_%d_pred", i*2+1);
		bpts_y.push_back(tree->Branch(key, &pred.at(i*2+1)));
	}

	for (Long64_t i = 0; i < tree->GetEntries(); i++) {
		if (i % 1000 == 0)
			std::printf("%d clusters processed\n", i);
		tree->GetEntry(i);
		std::vector<double> inputData = get_TTrained_input(inp);
		std::vector<double> predN = net->calculateNormalized(inputData);
		for (int j = 0; j < predN.size(); j++)
			pred.at(j) = predN.at(j);
		prepare_output(inp, predN, nparticles, residualsX, residualsY);
		for (int j = 0; j < nbins; j++) {
			bpts_x.at(j)->Fill();
			bpts_y.at(j)->Fill();
		}
	}

	out.Write(0, TObject::kWriteDelete);
}


