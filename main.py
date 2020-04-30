# !/usr/bin/env python3

import argparse
import configparser
import sys
import glob
import os
from time import time
import platform
import multiprocessing.dummy as mult_proc
from functools import partial


# modules
from PCprophet import io_ as io
from PCprophet import collapse as collapse
from PCprophet import generate_features as generate_features
from PCprophet import hypothesis as hypothesis
from PCprophet import map_to_database as map_to_database
from PCprophet import merge as merge
from PCprophet import differential as differential
from PCprophet import predict as predict
from PCprophet import plots as plots

from PCprophet import exceptions as exceptions
from PCprophet import validate_input as validate


class ParserHelper(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


# TODO check os
def get_os():
    return platform.system()


def create_config():
    '''
    parse command line and create .ini file for configuration
    '''
    parser = ParserHelper(description='Protein Complex Prophet argument')
    os_s = get_os()
    boolean = ['True', 'False']
    parser.add_argument(
        '-db',
        help='protein complex database from CORUM or ppi network in STRING format',
        dest='database',
        action='store',
        default='20190513_CORUMcoreComplexes.txt',
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
        '-Output',
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
        '-ppi',
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
        help='merge usign all complexes or reference only',
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
        default=0.5,
        type=float,
    )
    parser.add_argument(
        '-co',
        help='collapse mode',
        choices=['GO', 'CAL', 'SUPER', 'NONE'],
        dest='collapse',
        default='GO',
        action='store',
    )
    parser.add_argument(
        '-sc',
        help='score for missing proteins in differential analysis',
        dest='score_missing',
        action='store',
        default=0.5,
        type=float,
    )
    parser.add_argument(
        '-mult',
        help='Multi processing',
        dest='multi',
        action='store',
        default='False',
        choices=['True', 'False']
    )
    parser.add_argument('-w', dest='weight_pred', action='store', default=1, type=float)
    args = parser.parse_args()

    # create config file
    config = configparser.ConfigParser()
    config['GLOBAL'] = {
        'db': args.database,
        'sid': args.sample_ids,
        'go_obo': io.resource_path('go-basic.obo'),
        'sp_go': io.resource_path('tmp_GO_sp_only.txt'),
        'Output': args.out_folder,
        'cal': args.calibration,
        'mw': args.mwuni,
        'temp': r'./tmp',
        'mult': args.multi
    }
    config['PREPROCESS'] = {
        'is_ppi': args.is_ppi,
        'all_fract': args.all_fract,
        'merge': args.merge,
    }
    config['POSTPROCESS'] = {'fdr': args.fdr, 'collapse_mode': args.collapse}
    config['DIFFERENTIAL'] = {
        'score_missing': args.score_missing,
        'weight_pred': args.weight_pred,
        'fold_change': '-5,-2,2,5',
        'correlation': '0.3,0.9',
        'ratio': '-2,-0.5,0.5,2',
        'shift': '-10,-5,5,10',
        'weight_fold_change': 1,
        'weight_correlation': 0.75,
        'weight_ratio': 0.25,
        'weight_shift': 0.5,
    }

    # create config ini file for backup
    with open('ProphetConfig.conf', 'w') as conf:
        config.write(conf)
    return config


def preprocessing(infile, config):
     validate.InputTester(infile, 'in').test_file()
     map_to_database.runner(
         infile=infile,
         db=config['GLOBAL']['db'],
         is_ppi=config['PREPROCESS']['is_ppi'],
         use_fr=config['PREPROCESS']['all_fract'],
         )
     hypothesis.runner(
         infile=infile,
         hypothesis=config['PREPROCESS']['merge'],
         use_fr=config['PREPROCESS']['all_fract'],
         )
     # sample specific folder
     tmp_folder = io.file2folder(infile, prefix=config['GLOBAL']['temp'])
     merge.runner(base=tmp_folder, mergemode=config['PREPROCESS']['merge'])
     generate_features.runner(
         tmp_folder, config['GLOBAL']['go_obo'], config['GLOBAL']['sp_go']
         )
     predict.runner(tmp_folder)
     return True


# TODO fix mv resource_path to main for pynstaller
def main():
    config = create_config()
    validate.InputTester(config['GLOBAL']['db'], 'db').test_file()
    validate.InputTester(config['GLOBAL']['sid'], 'ids').test_file()
    files = io.read_sample_ids(config['GLOBAL']['sid'])
    files = [os.path.abspath(x) for x in files.keys()]
    if config['GLOBAL']['mult'] == 'True':
        p = mult_proc.Pool(len(files))
        preproc_conf=partial(preprocessing, config=config)
        p.map(preproc_conf, files)
        p.close()
        p.join()
    else:
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
        config['GLOBAL']['Output'],
        config['GLOBAL']['temp']
    )
    plots.runner(
        config['GLOBAL']['temp'],
        config['GLOBAL']['Output'],
        config['POSTPROCESS']['fdr'],
        config['GLOBAL']['sid'],
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
