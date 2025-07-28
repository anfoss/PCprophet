#!/usr/bin/env python3

import argparse
import configparser
import sys
import os
import platform
import numpy as np
from datetime import datetime
import pandas as pd

# modules
from PCprophet import io_ as io
from PCprophet import collapse as collapse
from PCprophet import features_and_prediction as features_and_prediction
from PCprophet import differential as differential
from PCprophet import generate_complexes as generate_complexes

from PCprophet import validate_input as validate


class ParserHelper(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


class Tee:
    def __init__(self, stream, logfile=None):
        self.stream = stream
        self.logfile = logfile

    def write(self, message):
        self.stream.write(message)
        self.stream.flush()
        if self.logfile:
            self.logfile.write(message)
            self.logfile.flush()


    def flush(self):
        self.stream.flush()
        if self.logfile:
            self.logfile.flush()


# TODO check os
def get_os():
    return platform.system()


def create_config():
    '''
    parse command line and create .ini file for configuration
    '''
    parser = ParserHelper(description='Protein Complex Prophet argument')
    parser.add_argument(
        '-db',
        help='protein complex database from CORUM or ppi network in STRING format',
        dest='database',
        action='store',
        default='meta/corum_allComplexes.txt',
    )
    # maybe better to add function for generating a dummy sample id?
    parser.add_argument(
        '-sid',
        help='sample ids file',
        dest='sample_ids',
        default='sample_ids.txt',
        action='store',
    )
    parser.add_argument(
        '-output',
        help='outfile folder path',
        dest='out_folder',
        default=r'./Output',
        action='store',
    )
    # TODO change tmp to Output/tmp check resource_path important for windows
    parser.add_argument(
        '-cal',
        help='calibration file no headers tab delimited fractiosn to mw in KDa',
        dest='calibration',
        default='None',
        action='store',
    )
    parser.add_argument(
        '-mw_uniprot',
        help='Molecular weight from uniprot',
        dest='mwuni',
        default='None',
        action='store',
    )
    parser.add_argument(
        '-is_ppi',
        help='is the -db a protein protein interaction database',
        dest='is_ppi',
        action='store',
        default='False',
        choices=['True', 'False'],
    )
    parser.add_argument(
        '-a',
        help='use all fractions [1,X]',
        dest='all_fract',
        action='store',
        default='all',
    )
    parser.add_argument(
        '-ma',
        help='merge using all complexes or reference only',
        dest='merge',
        action='store',
        choices=['all', 'reference'],
        default='all',
    )
    parser.add_argument(
        '-fdr',
        help='false discovery rate for novel complexes',
        dest='fdr',
        action='store',
        default=0.2,
        type=float,
    )
    parser.add_argument(
        '-co',
        help='collapse mode',
        choices=['GO', 'CAL', 'SUPER', 'PROB', 'NONE'],
        dest='collapse',
        default='GO',
        action='store',
    )
    parser.add_argument(
        '-dif',
        help='skip differential analysis',
        dest='dif',
        action='store',
        default=False,
    )
    parser.add_argument('-v', dest='verbose', help='Verbose', action='store', default=1)
    parser.add_argument('-skip',
                        dest='skip',
                        help='Skip feature generation and complex prediction step',action='store',
                        default=False)
    args = parser.parse_args()

    # deal with numpy warnings and so on
    if args.verbose == 0:
        np.seterr(all='ignore')
    else:
        pass
        # print them

    # create config file
    config = configparser.ConfigParser()
    config['GLOBAL'] = {
        'db': args.database,
        'sid': args.sample_ids,
        'go_obo': io.resource_path("meta/go_graph.graphml"),
        'sp_go': io.resource_path("meta/go_gaf.pkl"),
        'output': args.out_folder,
        'cal': args.calibration,
        'mw': args.mwuni,
        'temp': r'./tmp',
        'skip': args.skip,
        'diff': args.dif
    }
    config['PREPROCESS'] = {
        'is_ppi': args.is_ppi,
        'all_fract': args.all_fract,
        'merge': args.merge,
    }
    config['POSTPROCESS'] = {'fdr': args.fdr, 'collapse_mode': args.collapse}
    # create config ini file for backup
    with open('ProphetConfig.conf', 'w') as conf:
        config.write(conf)
    return config


def preprocessing(infile, config):
    #validate.InputTester(infile, 'in').test_file()
    generate_complexes.runner(
        infile=infile,
        db=config['GLOBAL']['db'],
        is_ppi=config['PREPROCESS']['is_ppi'],
        hypothesis=config['PREPROCESS']['merge'],
    )
    #  # sample specific folder
    tmp_folder = io.file2folder(infile, prefix=config['GLOBAL']['temp'])
    features_and_prediction.runner(
        base=tmp_folder,
        go_obo=config['GLOBAL']['go_obo'],
        tsp_go=config['GLOBAL']['sp_go'],
    )
    return True


def main():
    config = create_config()
    
    
    ## add logging
    log_out = f"log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log"
    log_file = open(log_out, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    validate.InputTester(config['GLOBAL']['db'], 'db').test_file()
    validate.InputTester(config['GLOBAL']['sid'], 'ids').test_file()
    files = pd.read_csv(config['GLOBAL']['sid'], sep='\t')
    files = [os.path.abspath(x) for x in files['Sample']]
    # skip feature generation
    if config['GLOBAL']['skip'] == 'False':
        [preprocessing(infile, config) for infile in files]
    collapse.runner(
        config['GLOBAL']['temp'],
        config['GLOBAL']['sid'],
        config['GLOBAL']['cal'],
        config['GLOBAL']['mw'],
        config['POSTPROCESS']['fdr'],
        config['POSTPROCESS']['collapse_mode'],
    )
    combined_file = os.path.join(config['GLOBAL']['temp'], 'combined.txt')
    differential.runner(
        combined_file,
        config['GLOBAL']['sid'],
        config['GLOBAL']['output'],
        config['GLOBAL']['temp'],
        config['GLOBAL']['diff'],
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
