import argparse
import logging
import os
import re
import subprocess

logging.basicConfig(level=logging.INFO)

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--ttrained', required=True)
p.add_argument('--output', required=True)
p.add_argument('--nparticles', type=int)
p.add_argument('--nbins', type=int, default=30)
args = p.parse_args()

if args.nparticles is None:
    m = re.match('.*\.pos([123]).*', args.input)
    if m is None:
        logging.error('unable to parse nparticles from input path, please specify --nparticles')
        exit(1)
    else:
        nparticles = int(m.group(1))
else:
    nparticles = args.nparticles

scriptdir = os.path.dirname(os.path.abspath(__file__))

if not os.path.isfile('{0}/error_NN_input'.format(scriptdir)):
    logging.error('{0}/error_NN_input: file not found'.format(scriptdir))
    exit(1)

logging.info('input: {0}'.format(args.input))
logging.info('ttrained: {0}'.format(args.ttrained))
logging.info('output: {0}'.format(args.output))
logging.info('nparticles: {0}'.format(nparticles))
logging.info('nbins: {0}'.format(args.nbins))
logging.info('launching error_NN_input')

subprocess.call([
    '{0}/error_NN_input'.format(scriptdir),
    args.input,
    args.ttrained,
    args.output,
    str(nparticles),
    str(args.nbins),
])
