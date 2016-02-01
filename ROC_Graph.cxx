#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <TFile.h>
#include <TGraph.h>
#include <vector>
using std::vector;

int main(int /*argc*/, char *argv[])
{

	TFile outfile(argv[2], "RECREATE");

	size_t len = 1024;
	char *lineptr = (char*)malloc(sizeof(char) * len);

	vector<double> TP;
	vector<double> FP;

	while (getline(&lineptr, &len, stdin) > -1) {

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


