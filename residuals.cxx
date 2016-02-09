// input format <localRowWeightedPosition> <localColumnWeightedPosition> <pitchesY>... <posX_truth> <posX_pred> <posY_truth> <posY_pred> ....

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <TFile.h>
#include <TH1D.h>

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

double get_pos(std::vector<double>& fields, size_t sizeY, bool truth, bool x, size_t ipart)
{
	size_t i_base = 2 + sizeY;
	if (!truth)
		i_base += 1;
	if (!x)
		i_base += 2;
	i_base += (4*(ipart-1));
	return fields.at(i_base);
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

std::vector<double> get_pitches(std::vector<double>& fields, size_t sizeY)
{
	std::vector<double> pitches;
	for (size_t i = 2; i < (2 + sizeY); i++)
		pitches.push_back(fields.at(i));
	return pitches;
}

int main(int argc, char *argv[])
{
	if (argc != 5) {
		std::cerr << "usage: " << argv[0] << " name out.root sizeY nparticles\n";
		return 1;
	}

	TFile outfile(argv[2], "RECREATE");
	size_t sizeY = atol(argv[3]);
	size_t nparticles = atol(argv[4]);

	std::string name(argv[1]);

	TH1D histX((name + "_X").c_str(), "", 1000, -0.05, 0.05);
	TH1D histY((name + "_Y").c_str(), "", 1000, -0.5, 0.5);

	std::string linebuf;
	int N = 0;
	while (std::getline(std::cin, linebuf)) {
	        std::vector<double> fields = parse_line(linebuf);
		
		if (fields.size() != ((nparticles * 4) + 2 + sizeY)) {
			std::cerr << "ERROR: malformed line\n";
			std::cerr << "       expected " << (nparticles*4 +2+sizeY) << " fields, received " << fields.size() << std::endl;
			std::cerr << "--> " << linebuf << std::endl;
			return 1;
		}
		double centerposX = fields.at(0);
		double centerposY = fields.at(1);
		std::vector<double> pitchesY = get_pitches(fields, sizeY);
		for (size_t i = 1; i <= nparticles; i++) {
			double pos_truth_idx_x = get_pos(fields, sizeY, true, true, i);
			double pos_predi_idx_x = get_pos(fields, sizeY, false, true, i);
			double pos_truth_x = correctedX(centerposX, pos_truth_idx_x);
			double pos_predi_x = correctedX(centerposX, pos_predi_idx_x);
			histX.Fill(pos_predi_x - pos_truth_x);
			double pos_truth_idx_y = get_pos(fields, sizeY, true, false, i);
			double pos_predi_idx_y = get_pos(fields, sizeY, false, false, i);
			double pos_truth_y = correctedY(centerposY, pos_truth_idx_y, sizeY, pitchesY);
			double pos_predi_y = correctedY(centerposY, pos_predi_idx_y, sizeY, pitchesY);
			histY.Fill(pos_predi_y - pos_truth_y);
		}
	}

	outfile.Write();

	return 0;
}
