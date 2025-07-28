import re
import pandas as pd
import numpy as np
import sys
import os
import networkx as nx
from datetime import datetime
from collections import defaultdict
import random
import time
import uuid
from pathlib import Path


# used in go_fdr.py
def create_file(filename, header):
    """
    create file in filename
    header is list
    """
    with open(filename, "w", encoding="utf-8") as outfile:
        outfile.write("%s\n" % "\t".join([str(x) for x in header]))

# used in go_fdr.py
def dump_file(filename, things):
    """
    dump things to file to filename
    """
    with open(filename, "a", encoding="utf-8") as outfile:
        outfile.write("%s\n" % things)


# used in generate_complexes.py
def file2folder(file_, prefix="./tmp/"):
    # we are already stripping the extension
    filename = os.path.splitext(os.path.basename(file_))[0]
    return os.path.join(prefix, filename)

# used everywhere
def resource_path(relative_path):
    """Resolve path to resource, working in dev and with PyInstaller."""
    try:
        # PyInstaller puts files in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Otherwise use project root = directory where main.py lives
        base_path = Path(sys.argv[0]).resolve().parent

    return base_path / relative_path


def catch(func, handle=lambda e: e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed
