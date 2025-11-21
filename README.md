# PCprophet

PCprophet is a software toolkit for protein complex prediction and differential analysis from cofractionation mass spectrometry (coFrac-MS) datasets.

## Getting Started

These instructions will guide you through obtaining a copy of the project, setting it up on your local machine, and testing compatibility with your current Python environment.

### Dependencies

Ensure that your environment satisfies the following dependencies:

- [Python >= 3.4](https://www.python.org)
- [scikit-learn >= 1.6.1](https://pypi.org/project/sklearn/)
- [NetworkX >= 3.3](https://networkx.org)
- [Pandas >= 2.2.2](https://pandas.pydata.org)
- [SciPy >= 1.13.1](https://www.scipy.org)
- [Dask >= 2025.4.1](https://dask.org)

### Installation

We recommend using [Anaconda](https://www.anaconda.com) to install and manage dependencies, as it simplifies setup for most operating systems.

#### On Windows

If you're having trouble adding Anaconda or Python to your PATH, refer to these resources:
*- [Installing Anaconda on Windows](https://www.datacamp.com/community/tutorials/installing-anaconda-windows)
*- [DLL load error fix for NumPy and SciPy](https://stackoverflow.com/questions/54063285/numpy-is-already-installed-with-anaconda-but-i-get-an-importerror-dll-load-fail)

## Usage

Refer to the [PCprophet_instructions.md](https://github.com/anfoss/PCprophet/blob/dev-branch/PCprophet_instructions.md) for a complete guide on how to prepare data, configure parameters, and run the full analysis pipeline.

### Mode Flag (`-mode`)

- `complex` (default): When `-is_ppi True`, clusters the input PPI network with Markov clustering to generate a complex database for complex-level prediction.
- `ppi`: Enables pairwise PPI prediction. PPI edge lists are used as-is (no clustering); complex databases and generated hypotheses are flattened to all pairwise interactions so you can run PPI scoring even when starting from a complex file. Deduplication collapses repeated pairs regardless of order.

Example: `python3 main.py -db myppi.txt -is_ppi True -mode ppi`

## License

This project is licensed under the MIT License – see the [LICENSE.md](LICENSE.md) file for details.

### Citation

If you use PCprophet in your work, please cite:

> Fossati A, Li C, Uliana F, Wendt F, Frommelt F, Sykacek P, Heusel M, Hallal M, Bludau I, Capraz T, Xue P, Song J, Wollscheid B, Purcell AW, Gstaiger M, Aebersold R (2021). PCprophet: a framework for protein complex prediction and differential analysis using proteomic data. *Nature Methods*, 13. https://doi.org/10.1038/s41592-021-01107-5

## Acknowledgments

- Thanks to [mojaje](https://github.com/mojaie/pygosemsim) for the original implementation of GO tree parsing.
