import argparse
import itertools as it
import os

import ROOT

dirpath = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.SetBatch()
ROOT.gROOT.LoadMacro('{}/AtlasStyle.C'.format(dirpath))
ROOT.gROOT.LoadMacro('{}/AtlasUtils.C'.format(dirpath))
ROOT.SetAtlasStyle()

def getArgs():
    p = argparse.ArgumentParser()
    p.add_argument('direction')
    p.add_argument('nparticles')
    p.add_argument('input1')
    p.add_argument('label1')
    p.add_argument('input2')
    p.add_argument('label2')
    return p.parse_args()


def graph(input1, input2, label1, label2, direction, nparticles):

    file1 = input1
    file2 = input2

    tfile1 = ROOT.TFile(file1, 'READ')
    tfile2 = ROOT.TFile(file2, 'READ')

    hist1 = tfile1.Get('error_all_all_pull')
    hist2 = tfile2.Get('error_all_all_pull')

    if hist2.Integral() == 0 or hist1.Integral() == 0:
 	return

    # hist1.Scale(1.0/hist1.Integral())
    # hist2.Scale(1.0/hist2.Integral())

    #assert hist1.GetEntries() == hist2.GetEntries()

    if nparticles == 2:
        hist1.Rebin(4)
        hist2.Rebin(4)
    elif nparticles == 3:
        hist1.Rebin(8)
        hist2.Rebin(8)

    c = ROOT.TCanvas('c', '', 0, 0, 600, 600)

    hist2.SetTitle(';pull; clusters')
    u = hist2.GetMean()
    s = hist2.GetStdDev()
    hist2.GetXaxis().SetRangeUser(u - 3*s, u + 3*s)
    hist2.SetLineColor(ROOT.kRed)
    hist2.SetStats()
    hist2.SetName(label2)
    hist2.GetYaxis().SetTitleOffset(1.5)
    hist2.SetMaximum(hist2.GetMaximum()*1.5)
    hist2.GetYaxis().SetLabelSize(0.035)
    hist2.Draw()

    hist1.SetLineColor(ROOT.kBlack)
    u = hist1.GetMean()
    s = hist1.GetStdDev()
    hist1.GetXaxis().SetRangeUser(u - 3*s, u + 3*s)
    hist1.SetStats()
    hist1.SetName(label1)
    hist1.Draw('sames')

    ROOT.gPad.Update()

    stats1 = hist1.FindObject('stats')
    stats1.SetX1NDC(0.75)
    stats1.SetX2NDC(0.95)
    stats1.SetY1NDC(0.75)
    stats1.SetY2NDC(0.85)

    stats2 = hist2.FindObject('stats')
    stats2.SetLineColor(ROOT.kRed)
    stats2.SetTextColor(ROOT.kRed)
    stats2.SetX1NDC(0.75)
    stats2.SetX2NDC(0.95)
    stats2.SetY1NDC(0.85)
    stats2.SetY2NDC(0.95)

    txt = ROOT.TText()
    txt.SetNDC()
    txt.SetTextSize(0.034)
    txt.DrawText(0.2,0.82, 'Local {0} direction'.format(direction))
    txt.DrawText(0.2,0.78, 'All layers')

    ROOT.ATLAS_LABEL(0.2,0.88)
    txt.DrawText(0.36, 0.88, "Internal")

    s = '' if nparticles == 1 else 's'
    txt.DrawText(0.2,0.74, 'JZ7 {} particle{} clusters'.format(nparticles,s))

    c.SaveAs('pull_{}_all_all_{}.pdf'.format(nparticles,direction))

def layer_name(l):
    if l == 'all':
        return 'All layers'
    if l == 'ibl':
        return 'IBL'
    if l == 'barrel':
        return 'Barrel'
    if l == 'endcap':
        return 'Endcap'

def main():
    args = getArgs()

    ROOT.gROOT.SetBatch()
    ROOT.SetAtlasStyle()
    ROOT.gStyle.SetOptStat(1101)
    ROOT.gStyle.SetStatBorderSize(1)
    ROOT.gStyle.SetStatFontSize(0.03)

    graph(args.input1, args.input2, args.label1, args.label2 ,args.direction,args.nparticles)

if __name__ == '__main__':
    main()
