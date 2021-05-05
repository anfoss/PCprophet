# PCprophet

Software toolkit for protein complex prediction and differential analysis of cofractionation mass spectrometry datasets.

## Getting Started

These instructions will guide you to obtain a copy of the project, to run on your local machine, and to test the compatibility with your current Python packages.
### Dependencies

* [Python >=3.4.x](https://www.python.org)
* [Sklearn 0.23.2](https://pypi.org/project/sklearn/)
* [NetworkX v2.4](https://networkx.github.io)
* [Pandas >0.23](https://pandas.pydata.org)
* [Scipy >1.5.2](https://www.scipy.org)

### Installing

We recommend using [anaconda](https://www.anaconda.com) as it contains the majority of the required packages for PCprophet. If you are using Windows and having problems adding paths of anaconda and Python, please click [here](https://www.datacamp.com/community/tutorials/installing-anaconda-windows) for guidance. Please also refer to this [page](https://stackoverflow.com/questions/54063285/numpy-is-already-installed-with-anaconda-but-i-get-an-importerror-dll-load-fail) for potential errors when importing python packages in Windows.

#### Command line version

Ensure that you have installed the GitHub tool and 'git-lfs' command specifically for large file transfer. Please see [here](https://gist.github.com/derhuerst/1b15ff4652a867391f03) for installing GitHub and [here](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) for the installing 'git-lfs' command.

```
git-lfs clone https://github.com/anfoss/PCprophet PCprophet
```
This will get you a working copy of PCprophet into a folder called PCprophet.

## Usage

For usage of PCprophet, refer to the [PCprophet_instructions.md](https://github.com/anfoss/PCprophet/blob/master/PCprophet_instructions.md).


## Contributing

Please read [CONTRIBUTING.md](https://github.com/anfoss/PCprophet/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Andrea Fossati** - *Initial work* - [anfoss](https://github.com/anfoss) andrea.fossati@ucsf.edu
* **Chen Li** - *Initial work* - chen.li@monash.edu


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

PCprophet citation:

Fossati, A., Li, C., Uliana, F., Wendt, F., Frommelt, F., Sykacek, P., Heusel, M., Hallal, M., Bludau, I., Capraz, T., Xue, P., Song, J., Wollscheid, B., Purcell, A. W., Gstaiger, M., & Aebersold, R. (2021). PCprophet: a framework for protein complex prediction and differential analysis using proteomic data. Nature Methods, 13. https://doi.org/10.1038/s41592-021-01107-5

## Acknowledgments

* [mojaje](https://github.com/mojaie/pygosemsim) for the implementation of GO tree parsing.
