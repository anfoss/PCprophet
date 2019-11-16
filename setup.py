# !/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name="PCProphet",
    version="1.0.dev",
    packages=find_packages(),
    scripts=[
        "collapse.py",
        "differential.py",
        "exceptions.py",
        "generate_features.py",
        "io_.py",
        "hypothesis.py",
        "mcl.py",
        "map_to_database.py",
        "parse_GO.py",
        "runner.py",
        #'plot_sec.py',
        "signal_prc.py",
        "stats_.py",
        "validate_input.py",
    ],
    # long_description=long_description,
    license="GNU General Public License v3.0",
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["scipy>=1.1", "pandas", "sklearn", "networkX"],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.obo"],
    },
    # metadata to display on PyPI
    author="Andrea Fossati",
    author_email="fossati@imsb.biol.ethz.ch",
    description="Prophet for analysis of co-fractionation data",
    keywords="Machine Learning Protein Complexes MS Fractionation",
    url="https://github.com/ChompOrDie/PCProphet",
    project_urls={
        "Bug Tracker": "https://github.com/fossatiA/PCProphet/",
        "Documentation": "https://github.com/fossatiA/PCProphet/",
        "Source Code": "https://github.com/fossatiA/PCProphet/",
    },
    classifiers=["License :: OSI Approved :: Python Software Foundation License"],
)
