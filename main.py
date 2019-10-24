# !/usr/bin/env python3

import argparse
import configparser
import sys
import glob
import os
from functools import wraps
from time import time
from subprocess import Popen
import platform

# modules
from PCProphet import io_ as io
from PCProphet import collapse as collapse
from PCProphet import generate_features as generate_features
from PCProphet import hypothesis as hypothesis
from PCProphet import map_to_database as map_to_database
from PCProphet import merge as merge
from PCProphet import differential_v4 as differential
from PCProphet import predict as predict
from PCProphet import plots as plots
# tests
from PCProphet import exceptions as exceptions
from PCProphet import validate_input as validate


class ParserHelper(argparse.ArgumentParser):
    """docstring for ParserHelper"""
    def error(self, message):
        sys.stderr.write('error: %s\n' %message)
        self.print_help()
        sys.exit(2)

def get_os():
  return platform.system()


def create_config():
    """
    parse command line and create .ini file for configuration
    """
    parser = ParserHelper(description='Protein Complex Prophet argument')
    os_s = get_os()
    boolean = ['True', 'False']
    parser.add_argument(
                        '-db',
                        dest='database',
                        action='store',
                        default='20190513_CORUMcoreComplexes.txt')
    # maybe better to add function for generating a dummy sample id?
    parser.add_argument(
                        '-sid',
                        help = 'sample ids file',
                        dest = 'sample_ids',
                        default = 'sample_ids.txt',
                        action = 'store')
    parser.add_argument(
                        '-Output',
                        help = 'outfile folder',
                        dest = 'out_folder',
                        default = r'./Output',
                        action = 'store')
    parser.add_argument(
                        '-cal',
                        help = 'cal file',
                        dest = 'calibration',
                        default = 'None',
                        action = 'store')
    parser.add_argument(
                        '-mw_uniprot',
                        help = 'Molecular weight from uniprot',
                        dest = 'mwuni',
                        default = 'None',
                        action = 'store')
    parser.add_argument('-ppi',
                        dest='is_ppi',
                        action='store',
                        default='False',
                        choices=['True', 'False']
                        )
    parser.add_argument('-a', dest='all_fract', action='store', default='all')
    parser.add_argument('-fh', dest='max_hypothesis',
                        action='store', default=50)
    parser.add_argument(
                        '-ma',
                        dest='merge',
                        action='store',
                        choices=['all', 'reference'],
                        default='all')
    parser.add_argument(
                        '-fdr',
                        dest='fdr',
                        action='store',
                        default=0.75,
                        type=float)
    parser.add_argument(
                        '-co',
                        help = 'collapse mode',
                        choices = ['GO', 'CAL', 'SUPER', 'NONE'],
                        dest = 'collapse',
                        default = 'GO',
                        action = 'store',
                        type = str.upper)
    parser.add_argument(
                        '-sc',
                        dest='score_missing',
                        action='store',
                        default=0.5,
                        type=float)
    parser.add_argument(
                        '-w',
                        dest='weight_pred',
                        action='store',
                        default=1,
                        type=float)
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
                        'temp': r'./tmp'
                        }
    config['PREPROCESS'] = {
                            'is_ppi': args.is_ppi,
                            'all_fract': args.all_fract,
                            'max_hypothesis': int(args.max_hypothesis),
                            'merge': args.merge
                            }
    config['POSTPROCESS'] = {
                            'fdr': args.fdr,
                            'collapse_mode': args.collapse}
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
                             'weight_shift': 0.5
                             }

    # create config ini file for backup
    with open('ProphetConfig.conf', 'w') as conf:
        config.write(conf)
    return config

# /Users/anfossat/Desktop/PCProphet_clean/ PCProphet
def main():
    config = create_config()
    validate.InputTester(config['GLOBAL']['db'], 'db').test_file()
    validate.InputTester(config['GLOBAL']['sid'],'ids').test_file()
    files = io.read_sample_ids(config['GLOBAL']['sid'])
    files = [os.path.abspath(x) for x in files.keys()]
    for infile in files:
        validate.InputTester(infile, 'in').test_file()
        map_to_database.runner(
                             infile=infile,
                             db=config['GLOBAL']['db'],
                             is_ppi=config['PREPROCESS']['is_ppi'],
                             use_fr=config['PREPROCESS']['all_fract'])
        hypothesis.runner(
                          infile=infile,
                          hypothesis=config['PREPROCESS']['merge'],
                          max_hypothesis=config['PREPROCESS']['max_hypothesis'],
                          use_fr=config['PREPROCESS']['all_fract'])
        # # #
        tmp_folder = io.file2folder(infile, prefix='./tmp/')
        merge.runner(
                     base=tmp_folder,
                     mergemode=config['PREPROCESS']['merge'])

        generate_features.runner(
                                 tmp_folder,
                                 config['GLOBAL']['go_obo'],
                                 config['GLOBAL']['sp_go'])
        # # assert False
        predict.runner(tmp_folder)
    collapse.runner(
                    config['GLOBAL']['temp'],
                    config['GLOBAL']['sid'],
                    config['GLOBAL']['cal'],
                    config['GLOBAL']['mw'],
                    config['POSTPROCESS']['fdr'],
                    config['POSTPROCESS']['collapse_mode'] )
    combined_file = os.path.join(config['GLOBAL']['temp'], 'combined.txt')
    ids = ['fold_change', 'correlation', 'ratio', 'shift']
    # keep it here for future implementation
    thresholds = [list(map(float, config['DIFFERENTIAL'][x].split(','))) for x in ids]
    desi_thresholds = dict(zip(ids, thresholds))
    weights = [float(config['DIFFERENTIAL']['weight_'+x]) for x in ids]
    desi_weights = dict(zip(ids, weights))
    differential.runner(
                        combined_file,
                        config['GLOBAL']['sid'],
                        config['GLOBAL']['Output'],
                        config['GLOBAL']['temp'],
                        config['DIFFERENTIAL']['score_missing'],
                        config['DIFFERENTIAL']['weight_pred'],
                        desi_thresholds,
                        desi_weights,
                        )
    plots.runner(
                config['GLOBAL']['temp'],
                config['GLOBAL']['Output'],
                config['POSTPROCESS']['fdr'],
                config['GLOBAL']['sid']
                )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
