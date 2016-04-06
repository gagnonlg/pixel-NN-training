import argparse
import itertools
import re

layers = [None, 'ibl', 'barrel', 'endcap']
eta_list = [None]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--type', choices=['number','pos1','pos2','pos3', 'error1x', 'error1y', 'error2x', 'error2y', 'error3x', 'error3y'], required=True)
    p.add_argument('--sizeY', default=7, type=int)
    p.add_argument('--nbins', type=int)
    return p.parse_args()

def name(layer, eta):

    if layer is None:
        layer = 'all'
    if eta is None:
        eta = 'all'
    else:
        eta = '-'.join(map(str,eta))

    return '_'.join([layer, eta])

def sql_where(truth, layer=None, eta=None):

    wlist = []

    if layer == 'ibl':
        wlist.append('(NN_barrelEC == 0 AND NN_layer == 0)')
    elif layer == 'barrel':
        wlist.append('(NN_barrelEC == 0 AND NN_layer > 0)')
    elif layer == 'endcap':
        wlist.append('abs(NN_barrelEC) == 2')

    if eta is not None:
        etamin, etamax = eta
        wlist.append('(abs(globalEta) >= %s AND abs(globalEta) <= %s)'
                     % (etamin,etamax))

    if isinstance(truth, str):
        truth = [truth]

    if len(truth) > 0:
        wlist.append('(' + ' OR '.join(map(lambda s: s + ' == 1', truth)) + ')')

    return ' AND '.join(wlist)

def sql_select(selected, truth=[], layer=None, eta=None):

    if isinstance(selected, str):
        selected = [selected]

    sql = 'SELECT ' + ','.join(selected) + ' FROM test '

    where = sql_where(truth, layer, eta)
    if where != '':
        sql += (' WHERE ' + where)

    return sql + ';'


def sql_number():

    ppairs = [(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)]

    for ((p,n),l,e) in itertools.product(ppairs, layers, eta_list):

        countp = sql_select(
            selected='count(*)',
            truth=('NN_nparticles%d_TRUTH' % p),
            layer=l,
            eta=e
        )

        countn = sql_select(
            selected='count(*)',
            truth=('NN_nparticles%d_TRUTH' % n),
            layer=l,
            eta=e
        )

        roc = sql_select(
            selected=['NN_nparticles%d_TRUTH' % p, 'NN_nparticles%d_PRED' % p],
            truth=['NN_nparticles%d_TRUTH' % p,'NN_nparticles%d_TRUTH' % n],
            layer=l,
            eta=e
        )

        roc = roc[:-1] + (' ORDER BY NN_nparticles%d_PRED DESC;' % p)

        qname = "ROC_%dvs%d_%s" % (p,n, name(l,e))

        print "%s|%s|%s|%s" % (qname, countp, countn, roc)

def sql_position(nparticles, sizeY):

    selected = ['NN_localEtaPixelIndexWeightedPosition','NN_localPhiPixelIndexWeightedPosition']
    for i in range(sizeY):
        selected.append('NN_pitches%d' % i)
    for i in range(nparticles):
        selected.append('NN_position_id_X_%d_TRUTH' % i)
        selected.append('NN_position_id_X_%d_PRED' % i)
        selected.append('NN_position_id_X_%d_TRUTH' % i)
        selected.append('NN_position_id_X_%d_PRED' % i)

    for l,e in itertools.product(layers,eta_list):
        qname = "residuals_%s" % name(l,e)
        sql = sql_select(
            selected=selected,
            layer=l,
            eta=e
        )

        print "%s|%s" % (qname,sql)

def sql_error(nparticles, sizeY, direction, nbins):

    selected = ['NN_localEtaPixelIndexWeightedPosition','NN_localPhiPixelIndexWeightedPosition']
    for i in range(sizeY):
        selected.append('NN_pitches%d' % i)
    for i in range(nparticles):
        selected.append('NN_position_id_%s_%d' % (direction,i))
        selected.append('NN_position_id_%s_%d_pred' % (direction,i))
    for i in range(nparticles):
        for j in range(nbins):
            selected.append('NN_error_%s_%d_%d_PRED' % (direction, i, j))

    for l,e in itertools.product(layers,eta_list):
        qname = "error_%s" % name(l,e)
        sql = sql_select(
            selected=selected,
            layer=l,
            eta=e
        )

        print "%s|%s" % (qname,sql)

if __name__ == '__main__':
    args = parse_args()
    if args.type == 'number':
        sql_number()
    elif args.type.startswith('pos'):
        sql_position(int(re.match('pos([123])', args.type).group(1)), args.sizeY)
    elif args.type.startswith('error'):
        m = re.match('error([123])([xy])', args.type)
        direction = m.group(2).upper()
        nparticles = int(m.group(1))

        if args.nbins == None:
            if nparticles == 1:
                nbins = 30
            elif nparticles == 2:
                nbins = 25
            else:
                nbins = 20
        else:
            nbins = args.nbins

        sql_error(nparticles, args.sizeY, direction, nbins)
