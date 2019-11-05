## Contributing to PCprophet


## I only have a question!

> **Note:** Please don't file an issue to ask a question

## What should I know before I get started?

### PCprophet modules

PCprophet is organized in a modular way. We have two kind of modules. First are core modules, which are characterized by the runner() method and that are called directly by the top level module main.py.
The others are module which have functions used by more than one module, or for which having a core module was unnecessary.

##### Core modules

* main.py - Top level module controlling the program flow and the setup of all parameters. Creates the .conf file
* map_to_database.py - Reads in the database (either PPI or complexes) and map the protein matrix into complexes. Performs rescaling and normalization and create the transf_matrix.txt file
* hypothesis.py - Performs hypothesis generation as described in the vignette and creates the transf_matrix_splitted.txt file
* merge.py - Merge hypothesis and database together before feature generation
* generate_feature.py - Generate features and creates the peak_list.txt file
* predict.py - Load pickled sklearn module and returns class probabiliy for every sample
* collapse.py - Experiment-wide and protein-centric merging of all complexes
* differential.py - Performs differential anslysi
* plots.py - Ensamble of plotting functions

#### Other modules

* aligner.py - Experiment-wide alignement using internal housekeeping complexes
* exceptions.py - Specific PCprophet extension
* go_fdr.py - GO-based FDR calculation as described in the vignette
* io_.py - I/O methods
* mcl.py - Markov based clustering for generating possible complexes from a ppi network
* parse_go.py - Implements GO based tree search and similarity methods calculation
* signal_prc.py - Toolset for signal processing
* stats_py - Frequently used statistical metrics
* validate_input.py - Input tester


## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for PCprophet.


> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check the [Vignette](https://discuss.atom.io/c/faq)** for a list of common questions and problems.


#### How Do I Submit A (Good) Bug Report?


Explain the problem and include details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Provide example of sample_ids.txt and a sample of the input matrix used**. Include links to files that can be useful for us to reproduce the error and to help us solving this issue faster
* **Report the error/exception raised and which parameters were used**
* **Explain which behavior you expected to see instead and why.**
* **Copy the .conf file and .spec file from PCprophet**

Provide more context by answering these questions:

* **Can you reproduce the problem in by running different files?**
* **Did the problem start happening recently** (e.g. after updating to a new version of Python or dependencies) or was this always a problem?
* If the problem started happening recently, **Can you reproduce the problem in a older environment**

Include details about your configuration and environment:

* **Which version of Python are you using?**

* **What's the name and version of the OS you're using**?
* **Are all the Python packages needed meeting the minimum version required?**

### How to propose a new feature

Our final goal is to allow biologists to have a quick and easy way to analyze co-fractionation data. We tried to implement most of the features we would like to have but of course something can always be added. Just contact [FossatiA](https://github.com/fossatiA) explaining

* Which feature would need to be added

* Which type of experiments will be possible to analyze thanks to that

* In which module it should be added

## Styleguides

For contributing we have the following code style for Python

### Python Styleguide

**We mostly refers to PEP8 for syntax and in general coding styles. Below are some of the major styleguides**

* Functions lowercase and _ are used when a space would be needed


```python
def do_something_important():
    """
    does something important
    """
    pass
```

* Classes definition in CamelCase


```python
class ImportantClass(object):
    """
    docstring for ImportantClass
    """
    def __init__(self, arg):
        super(ImportantClass, self).__init__()
        self.arg = arg
```


* Line length is 79 characters, apart when a long chained Pandas statement is used.

```python
(df.groupby[['x','y']]
.apply(lambda x: (np.max(x['z'])-np.min(x['z'])))
.sort_values(ascending=False))
```


* Functions that do not returns explicitly a value should return True
so they can be tested easily

```python
def do_something_important():
    """
    does something important
    """
    return True


def test_do_something_important():
    """
    test do something important
    """
    if do_something_important():
        pass
```
* Functions used only once in a specific module or function should be encapsulated within the function they are called in. We used both lambdas or functions depending on the context

```python
def do_something_important(df):
    """
    does something important
    """
    def does_something_important_but_only_here(things):
        """
        does something important but only here
        very long function
        """
        pass
    does_something_important_but_only_here()
    return True

```


### Documentation Styleguide

* Use [Markdown](https://daringfireball.net/projects/markdown)


### Code formatting Guidelines

* Use [Flake8](https://pypi.org/project/flake8/)
