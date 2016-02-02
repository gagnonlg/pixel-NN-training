#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <TFile.h>
#include <TGraph.h>
#include <vector>
using std::vector;

int main(int argc, char *argv[])
{
	if (argc < 3) {
		fprintf(stderr, "usage: %s <graph_name> <out.root> [fraction]\n", argv[0]);
		return 1;
	}

	float f = (argc >= 4)? atof(argv[3]) : 1;

	TFile outfile(argv[2], "RECREATE");

	size_t len = 1024;
	char *lineptr = (char*)malloc(sizeof(char) * len);

	vector<double> TP;
	vector<double> FP;

	size_t i = 0;

	while (getline(&lineptr, &len, stdin) > -1) {

		if (i++ % ((size_t)(1.0/f)) != 0)
			continue;

		double fp, tp;

		if (sscanf(lineptr, "%*s %lf %lf\n", &fp, &tp) != 2) {
			fprintf(stderr, "ERROR: %s: malformed line: %s\n", argv[0], lineptr);
			return 1;
		}

		FP.push_back(fp);
		TP.push_back(1 - tp);
	}

	TGraph roc(FP.size(), FP.data(), TP.data());

	roc.Write(argv[1]);
	outfile.Close();

	return 0;
}


