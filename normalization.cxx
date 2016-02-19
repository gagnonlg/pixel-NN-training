#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <gsl/gsl_rstat.h>

std::vector<float> parse_line(std::string& line)
{
	std::vector<float> parsed;
	size_t start = 0;
	size_t end = line.find(",", start);
	while (start != std::string::npos) {
		parsed.push_back(std::stof(line.substr(start, end - start)));
		start = (end == std::string::npos)? end : end + 1;
		end = line.find(",", start);
	}
	return parsed;
}


int main(int argc, char *argv[])
{
	std::vector<gsl_rstat_workspace*> acc;
	bool initialized = false;

	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " <in> <out>\n";
		return 1;
	}

	std::ifstream in(argv[1]);
        FILE *out = std::fopen(argv[2], "w");

	std::string line;
	bool first = true;

	while(std::getline(in, line)) {
		if (first) {
			first = false;
			continue;
		}
		std::vector<float> fields = parse_line(line);
		if (!initialized) {
		        for (size_t i = 0; i < fields.size(); i++)
				acc.push_back(gsl_rstat_alloc());
			initialized = true;
		}
		for (size_t i = 0; i < fields.size(); i++)
			gsl_rstat_add(fields.at(i), acc.at(i));
	}

	for (size_t i = 0; i < acc.size(); i++)
		fprintf(out, "%f ", gsl_rstat_mean(acc.at(i)));
	fprintf(out, "\n");
	for (size_t i = 0; i < acc.size(); i++)
		fprintf(out, "%f ", gsl_rstat_sd(acc.at(i)));
	fprintf(out, "\n");

	std::fclose(out);

	return 0;
}
