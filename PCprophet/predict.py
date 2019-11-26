# !/usr/bin/env python3


import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import joblib

import PCprophet.io_ as io


def deserialize(model):
    """
    return model
    """
    clf = joblib.load(model)
    return clf


def test_model():
    """
    test model score and check for os and load correct model
    """
    pass


def runner(base, model="./PCProphet/rf_equal.clf"):
    """
    get model file and run prediction
    """
    infile = os.path.join(base, "mp_feat_norm.txt")
    X, memo = io.prepare_feat(infile)
    clf = deserialize(model)
    prob = np.array(clf.predict_proba(X))
    pos = np.array(["Yes" if x == 1 else "No" for x in clf.predict(X)])
    out = np.concatenate((memo, prob, pos.reshape(-1, 1)), axis=1)
    header = ["ID", "NEG", "POS", "IS_CMPLX"]
    df = pd.DataFrame(out, columns=header)
    df = df[["ID", "POS", "NEG", "IS_CMPLX"]]
    outfile = os.path.join(base, "rf.txt")
    df.to_csv(outfile, sep="\t", index=False)
