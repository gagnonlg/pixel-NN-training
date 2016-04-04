import argparse
import logging
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--db', required=True)
p.add_argument('--output', required=True)
p.add_argument('--nparticles', type=int)
p.add_argument('--sizeY', type=int)
p.add_argument('--nbins', type=int, default=30)

if args.sizeY is None:
    m = re.match('.*[0-9]x([0-9]).*', args.input)
    if m is None:
        logging.error('unable to parse sizeY from input path, please specify --sizeY')
        exit(1)
    else:
        sizeY = int(m.group(1))
else:
    sizeY = args.sizeY

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

logging.info('launching error_NN_input')
logging.info('input: {0}'.format(args.input))
logging.info('db: {0}'.format(args.db))
logging.info('output: {0}'.format(args.output))
logging.info('nparticles: {0}'.format(nparticles))
logging.info('nbins: {0}'.format(args.nbins))
logging.info('sizeY: {0}'.format(sizeY))

subprocess.call([
    '{0}/error_NN_input',
    args.input,
    args.db,
    args.ouput,
    nparticles,
    args.nbins,
    sizeY
])
