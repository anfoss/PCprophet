# PCprophet

The software toolkit for protein complex prediction and differential analysis of cofractionation mass spectrometry datasets.

## Getting Started

These instructions will guide you to obtain a copy of the project, to run on your local machine, and to test the compatibility with your current Python packages.
### Dependencies

* [Python >=3.4.x](https://www.python.org)
* [Sklearn 0.20.3](https://pypi.org/project/sklearn/)
* [NetworkX v2.4](https://networkx.github.io)
* [Pandas >0.23](https://pandas.pydata.org)
* [Scipy >1.1.0](https://www.scipy.org)

### Installing

We recommend using [anaconda](https://www.anaconda.com) as it contains the majority of the required packages for PCprophet, while igraph needs to be installed separately [here](https://igraph.org/python/).

#### Command line version

Ensure that you have installed the GitHub tool and 'git-lfs' command specifically for large file transfer. Please see [here](https://gist.github.com/derhuerst/1b15ff4652a867391f03) for installing GitHub and [here](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) for the installing 'git-lfs' command.

```
git-lfs clone https://github.com/fossatiA/PCprophet PCprophet
```
This will get you a working copy of PCprophet into a folder called PCprophet.

## Usage

For usage of PCprophet, refer to the [PCprophet_instructions.md](https://github.com/fossatiA/PCprophet/blob/master/PCprophet_instructions.md).


## Contributing

Please read [CONTRIBUTE.md](https://github.com/fossatiA/PCprophet/blob/master/CONTRIBUTE.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Andrea Fossati** - *Initial work* - [fossatiA](https://github.com/fossatiA) fossati@imsb.biol.ethz.ch
* **Chen Li** - *Initial work* - chen.li@monash.edu


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* We applied the [mojaje](https://github.com/mojaie/pygosemsim) pacakge to the implementation of GO tree parsing.
