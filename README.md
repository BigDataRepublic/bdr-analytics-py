[![PyPI version](https://badge.fury.io/py/bdranalytics.svg)](https://badge.fury.io/py/bdranalytics)
[![Build Status](https://travis-ci.org/BigDataRepublic/bdr-analytics-py.svg?branch=master)](https://travis-ci.org/BigDataRepublic/bdr-analytics-py)

# BigData Republic Analytics (python)
Our analytics library to quickly get our data scientists up to speed, on the python platform

User documentation can be found at https://bigdatarepublic.github.io/bdr-analytics-py/

## Installation

Installation is done through the pip command line utility.

```
pip install bdranalytics
```

## Using the Spark notebooks
Some notebooks in the `notebooks` folder use spark. Check the [spark documentation](http://spark.apache.org/docs/2.0.1/programming-guide.html) for running jupyter with a spark contet.

But in short, for **windows**
```
set PYSPARK_DRIVER_PYTHON_OPTS=notebook
set PYSPARK_DRIVER_PYTHON=jupyter
[spark_install_dir]\bin\pyspark
```

And for **nix**
```
export PYSPARK_DRIVER_PYTHON_OPTS=notebook
export PYSPARK_DRIVER_PYTHON=jupyter
[spark_install_dir]/bin/pyspark
```

## Contributing
To contribute, please fork or branch from `master` and submit a pull-request.
Guidelines for an acceptable pull-request:

- PEP8 compliant code
- At least one line of documentation per class, function and method.
- Tests covering edge cases of your code.

### Development environment
To create the development environment with conda, run:

  > conda env create -f environment-dev.yml

  > source activate bdranalytics-dev

### Running the test

To run all tests:
> source activate bdranalytics-dev
> python setup.py test

### Creating a package dist

To create a dist from a local checkout (when developing on this module):
> source activate bdranalytics-dev
> python setup.py sdist

### Running the installation script
This uses the setup.py script directly, useful for testing how the dist will be installed without creating the dist.

To just install the package and main dependencies from a local checkout (when going to use this module):
> python setup.py install

### Creating the sphinx documentation

To update html files:
```
source activate bdranalytics-dev
cd doc
make clean && make source && make html
```
