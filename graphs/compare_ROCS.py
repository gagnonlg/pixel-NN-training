import argparse
import sys
import os
import re
import ROOT

dirpath = os.path.dirname(os.path.realpath(__file__))

ROOT.gROOT.SetBatch()
ROOT.gROOT.LoadMacro('{}/AtlasStyle.C'.format(dirpath))
ROOT.gROOT.LoadMacro('{}/AtlasUtils.C'.format(dirpath))
ROOT.SetAtlasStyle()

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs='*', required=True)
    p.add_argument("--labels", nargs='*')
    return p.parse_args(argv)

colors = [ROOT.kBlack, ROOT.kRed]

def layer_name(l):
    if l == 'all':
        return 'All layers'
    if l == 'ibl':
        return 'IBL'
    if l == 'barrel':
        return 'Barrel'
    if l == 'endcap':
        return 'Endcap'

def graph(key, grphs, labels):
    c = ROOT.TCanvas("c","c", 0, 0, 600, 600)
    c.SetLogx()
    leg = ROOT.TLegend(0.7,0.8,0.9,0.9)
    leg.SetBorderSize(0)
    first = True
    for g,col,lbl in zip(grphs,colors,labels):
        g.SetTitle(key + ";False positive rate;False Negative Rate;")
        g.SetLineColor(col)
        g.SetMarkerSize(0.05)
        g.SetMaximum(1.2)
        g.Draw('' if first else 'same')

        ROOT.ATLAS_LABEL(0.2,0.88)
        txt = ROOT.TText()
        txt.SetNDC()
        txt.SetTextSize(0.034)
        txt.DrawText(0.36, 0.88, "Internal")

        txt.SetTextSize(0.02)
        m = re.match('.*_([123])vs([123])_([a-zA-Z]+)', key)
        p = int(m.group(1))
        n = int(m.group(2))
        l = m.group(3)
        txt.DrawText(0.7,0.76, "JZ7 PIXEL clusters")
        txt.DrawText(0.7,0.73, layer_name(l))
        txt.DrawText(0.7,0.70, "True positive: {} particle{}".format(p, '' if p == 1 else 's'))
        txt.DrawText(0.7,0.67, "True negative: {} particle{}".format(n, '' if n == 1 else 's'))

        leg.AddEntry(g, lbl, "L")

        first = False

    leg.Draw()
    c.SaveAs(key + ".pdf")

def main(argv):
    args = parse_args(argv)
    labels = args.labels if args.labels is not None else map(str,range(len(args.inputs)))
    files = map(ROOT.TFile.Open, args.inputs)
    for k in map(lambda x: x.GetName(), files[0].GetListOfKeys()):
        graph(k,map(lambda f: f.Get(k), files), labels)
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
