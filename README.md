# PCprophet

Software toolkit for protein complex prediction and differential analysis of cofractionation mass spectrometry datasets

## Getting Started

These instructions will get you a copy of the project up and running on your local machine, how to test the compatibility with your current Python packages and

### Prerequisites

* [Python 3.x](https://www.python.org)
* [Sklearn 0.20.3](https://pypi.org/project/sklearn/)
* [NetworkX >2.1](https://networkx.github.io)
* [Pandas >0.23](https://pandas.pydata.org)
* [Scipy >1.1.0](https://www.scipy.org)
* [Java vxx](https://www.java.com)

### Installing

#### Command line version

For the command line version Java is not necessary and can be skipped.
```
github clone PCprophet
```

#### GUI version

Installing the GUI version of PCprophet can be done by either cloning the repo or by downloading the executable file from ### INSERT here link.

## Running the tests

The test module included with PCprophet tests most of the used function and the compatibility of the pickled model with the installed Sklearn version

### Test model


```
python3 PCprophet/test/test_model.py
```
PCprophet.Exceptions.ModelError will be raised in case of not compatibility and it will be necessary to downgrade the current Sklearn to a compatible version

### Test functions

```
python3 PCprophet/test/test_methods.py
```

PCprophet.Exceptions.MethodError will be raised in case of not compatibility and it will be necessary to downgrade the current Sklearn to a compatible version

## Usage

For usage of PCprophet refers to the vignette in ... link


## Built With

* [Pynstaller](https://www.pyinstaller.org) - Package builder
* [Maven](https://maven.apache.org/) - GUI


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Andrea Fossati** - *Initial work* - [fossatiA](https://github.com/fossatiA)
* **Chen Li** - *Initial work* - [fossatiA](https://github.com/fossatiA)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [mojaje](https://github.com/mojaie/pygosemsim) for the implementation of GO tree parsing
