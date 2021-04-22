# !/usr/bin/env python3

import os
from PCprophet import io_ as io
from PCprophet import collapse as collapse
from PCprophet import generate_features as generate_features
from PCprophet import hypothesis as hypothesis
from PCprophet import map_to_database as map_to_database
from PCprophet import merge as merge
from PCprophet import predict as predict
import main


def get_conf_files():
    test_ids = os.path.join('test', 'test_ids.txt')
    conf = main.create_config()
    conf['GLOBAL']['sid'] = test_ids
    files = io.read_sample_ids(conf['GLOBAL']['sid'])
    files = [os.path.abspath(x) for x in files.keys()]
    fl = files[0].split('/')[-1]
    tmp_f = os.path.join('tmp', fl.split('.')[0])
    return [conf, files[0], tmp_f]


def test_database():
    conf, fl, tmp_f = get_conf_files()
    fin = map_to_database.runner(
        infile=fl,
        db=conf['GLOBAL']['db'],
        is_ppi=conf['PREPROCESS']['is_ppi'],
        use_fr=conf['PREPROCESS']['all_fract'],
        )
    if not fin:
        assert False


def test_hypo():
    conf, fl, tmp_f = get_conf_files()
    fin = hypothesis.runner(
        infile=fl,
        hypothesis=conf['PREPROCESS']['merge'],
        use_fr=conf['PREPROCESS']['all_fract'],
        )
    if not fin:
        assert False


def test_gen_feat():
    conf, fl, tmp_f = get_conf_files()
    fin = generate_features.runner(
        tmp_f, conf['GLOBAL']['go_obo'], conf['GLOBAL']['sp_go']
        )
    if not fin:
        assert False


def test_predict():
    conf, fl, tmp_f = get_conf_files()
    fin = predict.runner(tmp_f)
    if not fin:
        assert False


def test_merge():
    conf, fl, tmp_f = get_conf_files()
    fin = merge.runner(base=tmp_f, mergemode=conf['PREPROCESS']['merge'])
    if not fin:
        assert False


def test_collapse():
    conf, fl, tmp_f = get_conf_files()
    fin = collapse.runner(
        conf['GLOBAL']['temp'],
        conf['GLOBAL']['sid'],
        conf['GLOBAL']['cal'],
        conf['GLOBAL']['mw'],
        conf['POSTPROCESS']['fdr'],
        conf['POSTPROCESS']['collapse_mode'],
    )
    if not fin:
        assert False


def main():

if __name__ == '__main__':
    main()
