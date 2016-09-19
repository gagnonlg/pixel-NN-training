import argparse
import os
import subprocess
import ROOT

ROOT.gROOT.SetBatch()
dirpath = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro('{}/AtlasStyle.C'.format(dirpath))
ROOT.gROOT.LoadMacro('{}/AtlasUtils.C'.format(dirpath))
ROOT.SetAtlasStyle()


def label(p, n):
    sp = '' if p == 1 else 's'
    sn = '' if n == 1 else 's'
    return '#splitline{Positive: %s particle clusters}{negative: %s particle clusters}' % (p,n)

def graph(thist_1, thist_2, true_pos, true_neg, layer, status, name):

    c = ROOT.TCanvas("c", "c", 0, 0, 800, 600)
    c.SetLogx()

    thist_1.SetTitle(';False positive rate;False Negative Rate;')
    thist_1.SetLineColor(ROOT.kBlack)
    thist_1.SetMarkerSize(0.05)
    thist_1.SetMaximum(1.7)
    #thist_1.GetXaxis().SetLimits(1e-4,1)
    thist_1.Draw()

    thist_2.SetLineColor(ROOT.kRed)
    thist_2.SetMarkerSize(0.05)
    thist_2.Draw('same')


    line = ROOT.TF1("f1", "1 - x", 0, 1)
    line.SetLineStyle(2)
    line.Draw('same')


    ROOT.ATLAS_LABEL(0.2,0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(0.32, 0.88, "Simulation {}".format(status))

    x_text = 0.20
    y_text = 0.70

    txt.SetTextSize(0.034)
    txt.DrawLatex(x_text, y_text + 0.08, 'PYTHIA8 dijet')
    txt.DrawLatex(x_text, y_text + 0.04, '1.8 < p_{T}^{jet} < 2.5 TeV')
    txt.DrawText(x_text, y_text, layer[1])

    leg = ROOT.TLegend(0.53,0.62,0.9,0.86)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.034)
    leg.AddEntry(thist_1, label(true_pos, true_neg), 'L')
    leg.AddEntry(thist_2, label(true_neg, true_pos), 'L')
    leg.AddEntry(line, "Random", 'L')
    leg.Draw()

    outname = '{}_ROC_{}_{}_{}.eps'.format(name, layer[0], true_pos, true_neg)
    c.SaveAs(outname)
    subprocess.call(['ps2pdf', '-dEPSCrop', outname])
    subprocess.call(['rm', outname])

import sys

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('path')
    args.add_argument('name')
    return args.parse_args()

def main():

    args = get_args()

    thinned = subprocess.check_output(['mktemp', '-u'])

    try:
        subprocess.check_call([
            '{}/thin_graph'.format(dirpath),
            args.path,
            thinned,
            '100'
        ])

        root_file = ROOT.TFile(thinned)

        for p,n in [(1,2), (1,3), (2,3)]:
            for l in ['all', 'ibl', 'barrel', 'endcap']:

                thist_1 = root_file.Get('ROC_{}vs{}_{}_all'.format(p,n,l))
                thist_2 = root_file.Get('ROC_{}vs{}_{}_all'.format(n,p,l))

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
                    true_pos=p,
                    true_neg=n,
                    layer=layers,
                    status='Internal',
                    name=args.name
                )

    finally:
        subprocess.call(['rm', '-f', thinned])

if __name__ == '__main__':
    main()
