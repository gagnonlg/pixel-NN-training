import argparse
import re
import os
from array import array
import ROOT

scriptdir = os.path.dirname(os.path.realpath(__file__))

ROOT.gROOT.SetBatch()
ROOT.gROOT.LoadMacro('{}/AtlasStyle.C'.format(scriptdir))
ROOT.gROOT.LoadMacro('{}/AtlasUtils.C'.format(scriptdir))

p = argparse.ArgumentParser()
p.add_argument('logfile')
p.add_argument('outname')
args = p.parse_args()

#regex = 'Epoch ([0-9]+): val_loss improved from .* to ([.0-9]+), saving model to '
regex = '[0-9]+s - loss: ([.0-9]+) - val_loss: ([.0-9]+)'

x = []
y = []
z = []

i = 0
with open(args.logfile, 'r') as logfile:
    for line in logfile.readlines():
        m = re.match(regex, line)
        if m is not None:
            x.append(i); i+=1;
            y.append(float(m.group(1)))
            z.append(float(m.group(2)))

ROOT.SetAtlasStyle()
c = ROOT.TCanvas('c','',600,600)
#g = ROOT.TGraph(len(x), array('d', x), array('d', y))
g = ROOT.TH1D('val_loss','', len(x), x[0], x[-1] + 1)
for b,v in zip(x,z):
    g.Fill(b,v)
g.SetTitle(';Epoch;Loss [Arb. U.]')

g.SetMaximum(g.GetMaximum() * 1.05)
g.GetYaxis().SetLabelSize(0.04)
g.GetYaxis().SetTitleOffset(1.6)
g.SetLineColor(ROOT.kRed)
g.Draw('HIST')

g2 = ROOT.TH1D('training_loss','', len(x), x[0], x[-1] + 1)
for b,v in zip(x,y):
    g2.Fill(b,v)

g2.Draw('HIST SAME')


leg = ROOT.TLegend(0.7,0.8,0.9,0.9)
leg.SetBorderSize(0)
leg.AddEntry(g2, 'Training loss', 'L')
leg.AddEntry(g, 'Validation loss', 'L')
leg.Draw()

txt = ROOT.TText()
txt.SetNDC()
txt.SetTextSize(0.04)
ROOT.ATLAS_LABEL(0.2,0.87)
txt.DrawText(0.37,0.87, 'Internal')

c.SaveAs(args.outname + ('.pdf' if not args.outname.endswith('.pdf') else ''))

