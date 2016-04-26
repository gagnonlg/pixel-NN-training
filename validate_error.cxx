// input format <localRowWeightedPosition> <localColumnWeightedPosition> <pitchesY>... <pos_truth> <pos_pred> .... <error bins>...

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>

std::vector<double> parse_line(std::string& line)
{
	std::vector<double> parsed;
	size_t start = 0;
	size_t end = line.find(" ", start);
	while (start != std::string::npos) {
		parsed.push_back(std::stod(line.substr(start, end - start)));
		start = (end == std::string::npos)? end : end + 1;
		end = line.find(" ", start);
	}
	return parsed;
}

std::pair<double,double> get_pos(std::vector<double>& fields, size_t start)
{
	return std::make_pair(fields.at(start), fields.at(start + 1));
}

double correctedX(double center_pos,
		  double pos_pixels,
		  double /*sizeY*/,
		  std::vector<double>& /*pitches*/)
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

std::vector<double> get_pitches(std::vector<double>& fields, size_t sizeY)
{
	std::vector<double> pitches;
	for (size_t i = 2; i < (2 + sizeY); i++)
		pitches.push_back(fields.at(i));
	return pitches;
}

std::vector<double> get_amplitudes(std::vector<double>& fields, size_t start)
{
	std::vector<double> amps;
	for (size_t i = start; i < fields.size(); i++)
		amps.push_back(fields.at(i));
	return amps;
}

std::vector<double> calculate_rms(std::vector<double> &output, int nparticles, double maximum)
{
	std::vector<double> output_rms;

	int dist_size = (int)output.size() / nparticles;

	double minimum = -maximum;

	for (int i=0; i < nparticles; i++)
	{
		double acc = 0;
		for (int u = 0; u < dist_size; u++)
			acc += output[i * dist_size + u];

		double rms = 0;
		for (int u = 0; u < dist_size; u++) {
			rms += output[i * dist_size + u] / acc * std::pow(minimum + (maximum - minimum)/(double)(dist_size - 2) * (u - 1./2.), 2);
		}
		rms = sqrt(rms);

		double interval = 3 * rms;

		//now recompute between -3*RMSx and +3*RMSx
		int min_bin = (int)(1+ (-interval - minimum) / (maximum - minimum) * (double)(dist_size - 2));
		int max_bin = (int)(1 +( interval - minimum) / (maximum - minimum) * (double)(dist_size - 2));

		if (max_bin > dist_size - 1)
			max_bin = dist_size - 1;
		if (min_bin < 0)
			min_bin = 0;

		rms = 0;
		for (int u = min_bin; u < max_bin + 1; u++)
			rms += output[i * dist_size + u] / acc * std::pow(minimum + (maximum - minimum)/(double)(dist_size - 2) * (u - 1./2.), 2);

		rms = sqrt(rms);

		output_rms.push_back(rms);
	}

	return output_rms;
}

int main(int argc, char *argv[])
{
	if (argc != 6) {
		std::cerr << "usage: " << argv[0] << " name out.root sizeY nparticles direction\n";
		return 1;
	}

	TFile outfile(argv[2], "RECREATE");
	size_t sizeY = atol(argv[3]);
	size_t nparticles = atol(argv[4]);
	char direction = argv[5][0];

	double maximum;
	if (direction == 'x') {
		maximum = (nparticles == 1)? 0.03 : 0.05;
	} else if (direction == 'y') {
		maximum = (nparticles == 1)? 0.3 : 0.4;
	} else {
		std::cerr << "unrecognized direction: " << direction << std::endl;
		return 1;
	}

	std::string name(argv[1]);

	TH1D hist_rms((name + "_rms").c_str(), "", 1000, -maximum, maximum);
	TH1D hist_res((name + "_res").c_str(), "", 1000, -maximum, maximum);
	TH1D hist_pull((name + "_pull").c_str(), "", 1000, -5, 5);
	TH2D hist_res_rms((name + "_res_rms").c_str(), "", 1000, -maximum, maximum, 1000, -maximum, maximum);

	std::string linebuf;
	int N = 0;
	while (std::getline(std::cin, linebuf)) {
	        std::vector<double> fields = parse_line(linebuf);

		if (fields.size() < ((nparticles * 2) + 2 + sizeY)) {
			std::cerr << "ERROR: malformed line\n";
			std::cerr << "       expected at least" << (nparticles*4 +2+sizeY) << " fields, received " << fields.size() << std::endl;
			std::cerr << "--> " << linebuf << std::endl;
			return 1;
		}
		double centerposX = fields.at(0);
		double centerposY = fields.at(1);
		double centerpos = (direction == 'x')? centerposX : centerposY;
		double (*corrected)(double, double, double, std::vector<double>&) = (direction == 'x')? correctedX : correctedY;


		std::vector<double> pitches = get_pitches(fields, sizeY);
		std::vector<double> amplitudes = get_amplitudes(fields, sizeY + 2 + 2*nparticles);
		std::vector<double> rms = calculate_rms(amplitudes, nparticles, maximum);

		for (size_t i = 0; i < nparticles; i++) {
			std::pair<double,double> pos = get_pos(fields, i*2 + sizeY + 2);
			double pos_truth = corrected(centerpos, pos.first, sizeY, pitches);
			double pos_predi = corrected(centerpos, pos.second, sizeY, pitches);
			double res = pos_predi - pos_truth;
			hist_rms.Fill(rms.at(i));
			hist_rms.Fill(res);
			hist_pull.Fill(res / rms.at(i));
			hist_res_rms.Fill(res, rms.at(i));
		}
	}

	outfile.Write();

	return 0;
}
