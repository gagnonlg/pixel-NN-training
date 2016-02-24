import argparse
import itertools
import re

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--type', choices=['number','pos1','pos2','pos3'], required=True)
    p.add_argument('--sizeY', default=7, type=int)
    return p.parse_args()

layers = [('all', None,None), ('ibl', 0,0), ('barrel', 1,0), ('endcap', None, 2)]

def sql_select(selected, truth=[], layer=None, barrel=None):

    if isinstance(selected, str):
        selected = [selected]
    if isinstance(truth, str):
        truth = [truth]

    if len(truth) > 0:
        where = ("(" + truth[0] + " == 1")
        for i in range(1, len(truth)):
            where += (" OR " + truth[i] + " == 1")
        where += ") "
    else:
        where = ""

    wherel = ""
    if layer is None and barrel == 2:
        wherel += " abs(NN_barrelEC) == 2"
    elif layer == 1 and barrel == 0:
        wherel += " NN_barrelEC == 0 AND NN_layer > 0"
    elif layer == 0 and barrel == 0:
        wherel += " NN_barrelEC == 0 AND NN_layer == 0"

    if where == "":
        where = wherel
    elif where != "" and wherel != "":
        where = where + " AND " + wherel

    sel = ""
    for s in selected:
        sel += (s + ",")
    sel = sel[:-1]

    sql = "SELECT " + sel + " FROM test"
    if where != "":
        sql += " WHERE " + where

    return sql + ";"


def sql_number():

    ppairs = [(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)]

    for ((p,n),(lname,l,b)) in itertools.product(ppairs, layers):

        countp = sql_select(
            selected='count(*)',
            truth=('NN_nparticles%d_TRUTH' % p),
            layer=l,
            barrel=b
        )

        countn = sql_select(
            selected='count(*)',
            truth=('NN_nparticles%d_TRUTH' % n),
            layer=l,
            barrel=b
        )

        roc = sql_select(
            selected=['NN_nparticles%d_TRUTH' % p, 'NN_nparticles%d_PRED' % p],
            truth=['NN_nparticles%d_TRUTH' % p,'NN_nparticles%d_TRUTH' % n],
            layer=l,
            barrel=b
        )

        roc = roc[:-1] + (' ORDER BY NN_nparticles%d_PRED DESC;' % p)

        name = "ROC_%dvs%d_%s" % (p,n,lname)

        print "%s|%s|%s|%s" % (name, countp, countn, roc)

def sql_position(nparticles, sizeY):

    selected = ['NN_localEtaPixelIndexWeightedPosition','NN_localPhiPixelIndexWeightedPosition']
    for i in range(sizeY):
        selected.append('NN_pitches%d' % i)
    for i in range(nparticles):
        selected.append('NN_position_id_X_%d_TRUTH' % i)
        selected.append('NN_position_id_X_%d_PRED' % i)
        selected.append('NN_position_id_X_%d_TRUTH' % i)
        selected.append('NN_position_id_X_%d_PRED' % i)

    for lname,l,b in layers:
        name = "residuals_%s" % lname
        sql = sql_select(
            selected=selected,
            layer=l,
            barrel=b
        )

        print "%s|%s" % (name,sql)

if __name__ == '__main__':
    args = parse_args()
    if args.type == 'number':
        sql_number()
    elif args.type.startswith('pos'):
        sql_position(int(re.match('pos([123])', args.type).group(1)), args.sizeY)
