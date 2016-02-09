import argparse
import re
import sys

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--sizeX", type=int, default=7)
    p.add_argument("--sizeY", type=int, default=7)
    p.add_argument("--type",  choices=['number','pos1','pos2','pos3'], required=True)
    p.add_argument("--old", action='store_true', default=False)
    return p.parse_args(argv)

def gen_inputs(sizeX, sizeY):
    fields = []
    for i in range(sizeX*sizeY):
        fields.append("NN_matrix%d" % i)
    for i in range(sizeY):
        fields.append("NN_pitches%d" % i)
    fields += [
        'NN_NN_layer',
        'NN_barrelEC',
        'NN_phi',
        'NN_theta'
    ]
    return fields

def gen_targets(type):
    fields = []
    if type.startswith('pos'):
        m = re.match('pos([123])', type)
        for i in range(int(m.group(1))):
            fields.append('NN_position_id_X_%d' % i)
            fields.append('NN_position_id_Y_%d' % i)
    elif type == 'number':
        fields.append('NN_nparticles1')
        fields.append('NN_nparticles2')
        fields.append('NN_nparticles3')
    return fields

def gen_metadata():
    fields = [
        'RunNumber',
        'EventNumber',
        'ClusterNumber',
        'NN_sizeX',
        'NN_sizeY',
        'NN_localEtaPixelIndexWeightedPosition',
        'NN_localPhiPixelIndexWeightedPosition',
        'NN_layer',
        'NN_barrelEC',
        'NN_etaModule',
        'NN_phi',
        'NN_theta',
    ]
    return fields

def main(argv):
    args = parse_args(argv)

    print "inputs:"
    for field in gen_inputs(args.sizeX,args.sizeY):
        print "  - %s" % field
    print "targets:"
    for field in gen_targets(args.type):
        print "  - %s" % field
    print "metadata:"
    for field in gen_metadata():
        print "  - %s" % field

    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
