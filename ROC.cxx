#include <float.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
	signal(SIGPIPE, SIG_DFL);

	if (argc != 3) {
		fprintf(stderr,
			"usage: %s <n. positives> <n. negatives>\n",
		        argv[0]);
		return 1;
	}

	double FP = 0, TP = 0;
	double fprev = DBL_MIN;

	double P = strtod(argv[1], NULL);
	double N = strtod(argv[2], NULL);

	size_t len = 2048;
	char *lineptr = (char*)malloc(sizeof(char) * len);

	printf("1 0 0\n");

	while (getline(&lineptr, &len, stdin) > -1) {
		double key;
		double f;
		if (sscanf(lineptr, "%lf %lf\n", &key, &f) != 2) {
			fprintf(stderr, "ERROR: %s: malformed line: %s\n", argv[0], lineptr);
			return 1;
		}

		if (f != fprev) {
			printf("%lf %lf %lf\n", f, FP / N, TP / P);
			fprev = f;
		}

		if (key == 1.0)
			++TP;
		else
			++FP;
	}

	printf("0 1 1\n");

	return 0;
}
