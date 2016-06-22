#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <TFile.h>
#include <TGraph.h>
#include <TObject.h>

int main(int argc, char *argv[])
{
	if (argc != 4) {
		printf("usage: %s in out nskip\n", argv[0]);
		return 1;
	}

	int nskip = std::atoi(argv[3]);

	TFile infile(argv[1], "READ");
	TFile outfile(argv[2], "CREATE");
	TIter next(infile.GetListOfKeys());
	TObject *obj;
	while ((obj = next())) {
		TObject *gobj = infile.Get(obj->GetName());
		if (std::strcmp(gobj->ClassName(),"TGraph") == 0) {
			TGraph *g = (TGraph*)gobj;
			size_t n = g->GetN() / nskip;
			double *gx = g->GetX();
			double *gy = g->GetY();
			double x[n];
			double y[n];
			for (size_t i = 0; i < n; i++) {
				x[i] = gx[i*nskip];
				y[i] = gy[i*nskip];
			}
			TGraph thinned(n,x,y);
			thinned.Write(obj->GetName());
		}
	}
	outfile.Close();
}
