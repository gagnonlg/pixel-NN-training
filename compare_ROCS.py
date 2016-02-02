import argparse
import sys
import ROOT

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs='*', required=True)
    p.add_argument("--labels", nargs='*')
    return p.parse_args(argv)

colors = [ROOT.kBlack, ROOT.kRed]

def graph(key, grphs, labels):
    c = ROOT.TCanvas("c","c", 0, 0, 800, 600)
    c.SetLogx()
    leg = ROOT.TLegend(0.7,0.8,0.9,0.9)
    first = True
    for g,col,lbl in zip(grphs,colors,labels):
        if first:
            g.SetTitle(key + ";False positive rate;False Negative Rate;")
            g.SetLineColor(col)
            g.Draw()
            first = False
        else:
            g.SetLineColor(col)
            g.Draw('same')

        leg.AddEntry(g, lbl, "L")

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
