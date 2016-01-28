from copy import deepcopy
import itertools

template_ROC   = "SELECT %s FROM test WHERE %s ORDER BY NN_nparticles%d_PRED DESC"
template_COUNT = "SELECT count(*) FROM test WHERE (%s) AND NN_nparticles%d_TRUTH == 1"

particle_pairs = [(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)]
layers_barrelEC = [(None,None), (0,0), (1,0), (None, 2)]


prod = itertools.product(
    particle_pairs,
    layers_barrelEC
)


for (p,n),(l,b) in prod:

    output  = "NN_nparticles%d_TRUTH,NN_nparticles%d_PRED" % (p,p)
    where   = "NN_nparticles%d_TRUTH == 1 OR NN_nparticles%d_TRUTH == 1 " % (p,n)

    if l is None and b == 2:
        where += " AND abs(NN_barrelEC) == 2"
        lname = "endcap"
    elif l == 1 and b == 0:
        where += " AND NN_barrelEC == 0 AND NN_layer > 0"
        lname = "barrel"
    elif l == 0 and b == 0:
        where += " AND NN_barrelEC == 0 AND NN_layer == 0"
        lname = "ibl"
    elif l is None and b is None:
        lname = "all"
        pass

    sql_ROC     = template_ROC % (output, where, p)
    sql_COUNT_p = template_COUNT % (where, p)
    sql_COUNT_n = template_COUNT % (where, n)

    name = "ROC_%dvs%d_%s" % (p,n,lname)

    print "%s|%s|%s|%s" % (name, sql_COUNT_p, sql_COUNT_n, sql_ROC)

