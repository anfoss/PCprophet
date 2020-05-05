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
    # create config file
    config = configparser.ConfigParser()
    config.read('ProphetConfig.conf')
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
