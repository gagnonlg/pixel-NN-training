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


def graph(thist_1, thist_2, thist_3, nparticles, direction, status, name):

    c = ROOT.TCanvas("c", "c", 0, 0, 800, 600)

    p = int(nparticles.split(' ')[0])

    if p == 1:
        rebin = 1
        f = 1.2
    elif p == 2:
        rebin = 4
        f = 5.5
    elif p == 3:
        rebin = 10
        f = 15


    normalize(thist_1)
    normalize(thist_2)
    normalize(thist_3)

    if direction == 'X':
        xmax = 0.04
        ymax = thist_1.GetMaximum()
    else:
        xmax = 0.4
        ymax = thist_2.GetMaximum()
        #f *= 1.5


    thist_1.SetTitle(';Truth hit residuals [mm]; Cluster density;')
    thist_1.SetLineColor(ROOT.kBlack)
    # thist_1.SetMarkerSize(0.05)
    # thist_1.SetMaximum(1.4)
    thist_1.Rebin(rebin)
    thist_1.GetXaxis().SetRangeUser(-xmax,xmax)
    thist_1.SetMaximum(ymax*f)

    thist_1.Draw()

    thist_2.SetLineColor(ROOT.kRed)
    thist_2.SetMarkerSize(0.05)
    thist_2.Rebin(rebin)
    thist_2.Draw('same')

    thist_3.SetLineColor(ROOT.kBlue)
    thist_3.SetMarkerSize(0.05)
    thist_3.Rebin(rebin)
    thist_3.Draw('same')

    ROOT.ATLAS_LABEL(0.2,0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(0.32, 0.88, "Simulation {}".format(status))
    #txt.DrawText(0.2, 0.84, status)

    txt.SetTextSize(0.034)
    txt.DrawLatex(0.19, 0.8, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
    txt.DrawText(0.19, 0.76, nparticles)
    txt.DrawText(0.19, 0.72, 'local {} direction'.format(direction))

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


    mu_1 = round(mu_1, 2)
    mu_2 = round(mu_2, 2)
    mu_3 = round(mu_3, 2)

    if mu_1 == -0.0:
        mu_1 = 0
    if mu_2 == -0.0:
        mu_2 = 0
    if mu_3 == -0.0:
        mu_3 = 0

    txt.SetTextColor(ROOT.kBlack)
    txt.DrawLatex(0.19, 0.66, '#mu = %.2f, #sigma = %.2f' % (mu_1, si_1))
    txt.SetTextColor(ROOT.kRed)
    txt.DrawLatex(0.19, 0.62, '#mu = %.2f, #sigma = %.2f' % (mu_2, si_2))
    txt.SetTextColor(ROOT.kBlue)
    txt.DrawLatex(0.19, 0.58, '#mu = %.2f, #sigma = %.2f' % (mu_3, si_3))

    leg = ROOT.TLegend(0.66,0.75,0.9,0.9)
    leg.SetBorderSize(0)
    leg.AddEntry(thist_1, 'Barrel clusters', 'L')
    leg.AddEntry(thist_2, 'IBL-only clusters', 'L')
    leg.AddEntry(thist_3, 'Endcap clusters', 'L')
    leg.Draw()

    c.SaveAs('{}_residuals_{}_{}.pdf'.format(name, p,direction))

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('path')
    args.add_argument('name')
    return args.parse_args()

def main():
    args = get_args()
    for i in range(1,4):
        for d in ['X', 'Y']:
            root_file = ROOT.TFile(args.path)
            thist_1 = root_file.Get('residuals_barrel_all_{}'.format(d))
            thist_2 = root_file.Get('residuals_ibl_all_{}'.format(d))
            thist_3 = root_file.Get('residuals_endcap_all_{}'.format(d))
            graph(
                thist_1=thist_1,
                thist_2=thist_2,
                thist_3=thist_3,
                nparticles='{} particle clusters'.format(i),
                direction=d,
                status='Internal',
                name=args.name
            )

if __name__ == '__main__':
    main()
