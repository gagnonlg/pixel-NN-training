#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <sqlite3.h>


class ErrorTree  {
public:
	TTree *tree;
	int nparticles;
	int nbins;
	int sizeY;
	std::vector<std::vector<double>*> *residualsX;
	std::vector<std::vector<double>*> *residualsY;

	double RunNumber;
	double EventNumber;
	double ClusterNumber;
	double NN_localEtaPixelIndexWeightedPosition;
	double NN_localPhiPixelIndexWeightedPosition;
	std::vector<double> pitches;

	ErrorTree(TTree *orig, int npart, int nbin, int size_Y)
	{
		tree = (TTree*)orig->CloneTree();
		nparticles = npart;
		nbins = nbin;
		sizeY = size_Y;
		residualsX = new std::vector<std::vector<double>*>;
		residualsY = new std::vector<std::vector<double>*>;
		for (int i = 0; i < nparticles; i++) {
			residualsX->push_back(new std::vector<double>(nbins));
			residualsY->push_back(new std::vector<double>(nbins));
			for (int j = 0; j < nbins; j++) {
				char key[256];
				std::sprintf(key, "NN_ERROR_X_%d_%d", i, j);
				tree->Branch(key, &(residualsX->at(i)->at(j)));
				std::sprintf(key, "NN_ERROR_Y_%d_%d", i, j);
				tree->Branch(key, &(residualsY->at(i)->at(j)));
			}
		}
#define CONNECT(b) tree->SetBranchAddress(#b, &b)
		CONNECT(RunNumber);
		CONNECT(EventNumber);
		CONNECT(ClusterNumber);
		CONNECT(NN_localEtaPixelIndexWeightedPosition);
		CONNECT(NN_localPhiPixelIndexWeightedPosition);
#undef CONNECT

		pitches.resize(sizeY);
		for (int i = 0; i < sizeY; i++) {
			char key[256];
			std::sprintf(key, "NN_pitches%d", i);
			tree->SetBranchAddress(key, &(pitches[i]));
		}
	}
};


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


int callback(void *data, int ncol, char **values, char **names)
{
	ErrorTree *obj = (ErrorTree*)data;

	std::map<std::string,double> map;
	for (int i = 0; i < ncol; i++)
		map[std::string(names[i])] = std::atof(values[i]);

	double centerposX = obj->NN_localEtaPixelIndexWeightedPosition;
	double centerposY = obj->NN_localPhiPixelIndexWeightedPosition;

	for (int i = 0; i < obj->nparticles; i++) {
		char key[256];
		std::sprintf(key, "NN_position_id_X_%d_PRED", i);
		double x_pred = map[key];
		std::sprintf(key, "NN_position_id_Y_%d_PRED", i);
		double y_pred = map[key];
		std::sprintf(key, "NN_position_id_X_%d_TRUTH", i);
		double x_truth = map[key];
		std::sprintf(key, "NN_position_id_Y_%d_TRUTH", i);
		double y_truth = map[key];

		x_pred = correctedX(centerposX, x_pred);
		x_truth = correctedX(centerposX, x_truth);

		y_pred = correctedY(centerposY,
				    y_pred,
				    obj->sizeY,
				    obj->pitches);
		y_truth = correctedY(centerposY,
				     y_truth,
				     obj->sizeY,
				     obj->pitches);


		/* X */
		double max = (i == 0)? 0.03 : 0.05;
		fill_amplitude(x_pred - x_truth, obj->residualsX->at(i), max);
		/* Y */
		max = (i == 0)? 0.3 : 0.5;
		fill_amplitude(x_pred - x_truth, obj->residualsY->at(i), max);
	}
	obj->tree->Fill();

	return 0;
}


std::string create_sql_query(int nparticles,
			     double RunNumber,
			     double EventNumber,
			     double ClusterNumber)
{
	std::ostringstream sql;

	sql << "SELECT ";
	for (int i = 0; i < nparticles; i++) {
		sql << "NN_position_id_X_" << i << "_TRUTH,";
		sql << "NN_position_id_X_" << i << "_PRED,";
		sql << "NN_position_id_Y_" << i << "_TRUTH,";
		sql << "NN_position_id_Y_" << i << "_PRED";
		if (i < nparticles - 1)
			sql << ',';
	}
	sql << " FROM test WHERE ";
	sql << "RunNumber=" << RunNumber
	    << " AND "
	    << "EventNumber=" << EventNumber
	    << " AND "
	    << "ClusterNumber=" << ClusterNumber
	    << ";";

	// std::puts(sql.str().c_str());

	return sql.str();
}


int main(int argc, char *argv[])
{
	if (argc != 7) {
		std::fprintf(stderr, "usage: %s <inrootfile> <db> <outrootfile> <nparticles> <nbins> <sizeY>", argv[0]);
		return 1;
	}

	TFile in(argv[1], "READ");
	TTree *orig = (TTree*)in.Get("NNinput");

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;
	rc = sqlite3_open(argv[2], &db);
	if (rc) {
		std::fprintf(stderr, sqlite3_errmsg(db));
		return 1;
	}

	/* This line speeds up the code by a lot */
	sqlite3_exec(db, "CREATE INDEX ClusterID IF NOT EXISTS ON test (RunNumber, EventNumber, ClusterNumber", NULL, NULL, NULL);

	TFile out(argv[3], "RECREATE");

	ErrorTree obj(orig,
		      std::atoi(argv[4]),
		      std::atoi(argv[5]),
		      std::atoi(argv[6]));

	for (Long64_t i = 0; i < obj.tree->GetEntries(); i++) {
		obj.tree->GetEntry(i);
		std::string sql = create_sql_query(obj.nparticles,
						   obj.RunNumber,
						   obj.EventNumber,
						   obj.ClusterNumber);

		rc = sqlite3_exec(db, sql.c_str(), callback, (void*)&obj, &zErrMsg);
		if (rc) {
			std::fprintf(stderr, "sql error: %s\n", zErrMsg);
			return 1;
		}
		if (i % 1000 == 0)
			std::printf("error_NN_input: processed %d clusters\n", i+1);
	}

	out.Write(0, TObject::kWriteDelete);
}


