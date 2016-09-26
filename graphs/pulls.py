import argparse
import os
import ROOT

ROOT.gROOT.SetBatch()
dirpath = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro('{}/AtlasStyle.C'.format(dirpath))
ROOT.gROOT.LoadMacro('{}/AtlasUtils.C'.format(dirpath))
ROOT.SetAtlasStyle()

def normalize(h):
    h.Scale(1.0/h.Integral())


def graph(thist_1, thist_2, thist_3, layers, direction, status, name):

    c = ROOT.TCanvas("c", "c", 0, 0, 800, 600)

    normalize(thist_1)
    normalize(thist_2)
    normalize(thist_3)

    gaus_1 = ROOT.TF1("f1", "gaus", -5, 5);
    fit_1 = thist_1.Fit("f1", "0")
    mu_1 = gaus_1.GetParameter(1)
    si_1 = gaus_1.GetParameter(2)

    gaus_2 = ROOT.TF1("f2", "gaus", -5, 5);
    fit_2 = thist_2.Fit("f2", "0")
    mu_2 = gaus_2.GetParameter(1)
    si_2 = gaus_2.GetParameter(2)

    gaus_3 = ROOT.TF1("f3", "gaus", -5, 5);
    fit_3 = thist_3.Fit("f3", "0")
    mu_3 = gaus_3.GetParameter(1)
    si_3 = gaus_3.GetParameter(2)

    if layers[0] == 'endcap':
        thist_1.Rebin(10)
        thist_2.Rebin(10)
        thist_3.Rebin(10)
    elif layers[0] == 'ibl':
        thist_1.Rebin(4)
        thist_2.Rebin(4)
        thist_3.Rebin(4)
    else:
        thist_1.Rebin(2)
        thist_2.Rebin(2)
        thist_3.Rebin(2)

    if direction == 'X':
        xmax = 0.04
        ymax = thist_1.GetMaximum() * 1.5
        f = 1
    else:
        xmax = 0.4
        ymax = thist_2.GetMaximum()
        f = 1.41

    if layers[0] == 'endcap':
        f *= 1.2


    thist_1.SetTitle(';Pull; Cluster density;')
    thist_1.SetLineColor(ROOT.kBlack)
    # thist_1.SetMarkerSize(0.05)
    # thist_1.SetMaximum(1.4)
    #thist_1.Rebin(rebin)
    #thist_1.GetXaxis().SetRangeUser(-xmax,xmax)
    thist_1.SetMaximum(ymax*f)


    thist_1.Draw()



    thist_2.SetLineColor(ROOT.kRed)
    thist_2.SetMarkerSize(0.05)
    #thist_2.Rebin(rebin)
    thist_2.Draw('same')

    thist_3.SetLineColor(ROOT.kBlue)
    thist_3.SetMarkerSize(0.05)
    #thist_3.Rebin(rebin)
    thist_3.Draw('same')

    ROOT.ATLAS_LABEL(0.2,0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(0.05)
    txt.DrawText(0.32, 0.88, "Simulation {}".format(status))
    #txt.DrawText(0.2, 0.84, status)

    txt.SetTextSize(0.034)
    txt.DrawLatex(0.19, 0.8, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
    txt.DrawText(0.19, 0.76, layers[1])
    txt.DrawText(0.19, 0.72, 'local {} direction'.format(direction))

    txt.SetTextColor(ROOT.kBlack)
    txt.DrawLatex(0.19, 0.66, '#mu = %.2f, #sigma = %.2f' % (mu_1, si_1))
    txt.SetTextColor(ROOT.kRed)
    txt.DrawLatex(0.19, 0.62, '#mu = %.2f, #sigma = %.2f' % (mu_2, si_2))
    txt.SetTextColor(ROOT.kBlue)
    txt.DrawLatex(0.19, 0.58, '#mu = %.2f, #sigma = %.2f' % (mu_3, si_3))


    leg = ROOT.TLegend(0.66,0.75,0.9,0.9)
    leg.SetBorderSize(0)
    leg.AddEntry(thist_1, '1-particle clusters', 'L')
    leg.AddEntry(thist_2, '2-particle clusters', 'L')
    leg.AddEntry(thist_3, '3-particle clusters', 'L')
    leg.Draw()

    c.SaveAs('{}_error_{}_{}.pdf'.format(name, direction,layers[0]))


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('path_1')
    args.add_argument('path_2')
    args.add_argument('path_3')
    args.add_argument('name')
    args.add_argument('direction')
    return args.parse_args()

def main():
    args = get_args()

    for l in ['all', 'ibl', 'endcap', 'barrel']:
        root_file_1 = ROOT.TFile(args.path_1)
        thist_1 = root_file_1.Get('error_{}_all_pull'.format(l))
        root_file_2 = ROOT.TFile(args.path_2)
        thist_2 = root_file_2.Get('error_{}_all_pull'.format(l))
        root_file_3 = ROOT.TFile(args.path_3)
        thist_3 = root_file_3.Get('error_{}_all_pull'.format(l))

        if l == 'all':
            layers=('all', 'All PIXEL layers')
        elif l == 'ibl':
            layers=('ibl', 'IBL')
        elif l == 'endcap':
            layers=('endcap', 'Endcaps')
        else:
            layers=('barrel', 'Barrel')

        graph(
            thist_1=thist_1,
            thist_2=thist_2,
            thist_3=thist_3,
            layers=layers,
            direction=args.direction,
            status='Internal',
            name=args.name
        )


if __name__ == '__main__':
    main()
