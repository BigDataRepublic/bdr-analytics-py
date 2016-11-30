# BigData Republic Analytics (python)
Our analytics library to quickly get our data scientists up to speed, on the python platform

## To use this library

There are different ways to include this package in your environment:

### PIP Without git checkout
This is a simple way to use this package. Especially if you do not plan to edit it during your current project.

To install the package without checking it out (to use this module):
  * First check `requirements-dev.txt` and install the correct requirements (like sklearn), including their dependencies.
  * Then install the bdr-analytics-py including dependencies:

    > pip install -e git+ssh://git@github.com/BigDataRepublic/bdr-analytics.git#egg=bdranalytics-0.1
    
### PIP With git checkout
This first downloads the package, and then installs it. This is especially useful if you want to make some changes in this package during your project.

To install the package including both its requiremets and dependencies:
  * First checkout from github

    > git clone git@github.com:BigDataRepublic/bdr-analytics-py.git

    > cd bdr-analytics-py
  * Then install bdr-analytics-py including requirements and dependencies:

    > pip install -r requirements.txt
    
### Conda without git checkout
This is another simple way to use this package. Especially if you do not plan to edit it during your current project.

To install the package without checking it out (to use this module):
  * First check `environment.yml` and install the correct requirements (like sklearn), including their dependencies.
  * Then install the bdr-analytics-py including dependencies:

    > pip install -e git+ssh://git@github.com/BigDataRepublic/bdr-analytics.git#egg=bdranalytics-0.1

### Conda with git checkout: Recommended
This is the recommended way, as all dependencies are automatically installed.

This first downloads the package, and then installs it. This is especially useful if you want to make some changes in this package during your project.
To install the package including both its requiremets and dependencies:
  * First checkout from github

    > git clone git@github.com:BigDataRepublic/bdr-analytics-py.git

    > cd bdr-analytics-py
    
  * For **existing** conda env 
   
    > conda env update --name=bdranalytics -f environment.yml
    
    Assuming you already have an environment named `bdranalytics`
    
  * For **new** conda env 
  
    > conda env create -f environment.yml
    
    This will create an environment named `bdranalytics`
    
## A more elaborate setup description

If you need an environment including this module, the recommended way to go is:

  1. Get the most recent codebase (including concrete requirements):
  > git clone git@github.com:BigDataRepublic/bdr-analytics-py.git

  > cd bdr-analytics-py
  
  2.  For **pip**: Optional but recommended, create a virtual env for your work:
  > pyenv install 2.7.11

  > pyenv virtualenv 2.7.11 bdranalytics

  > pyenv activate bdranalytics

  > pip install --upgrade pip
  
  Install the module, it's dependencies and requirements.
  
  > pip install -r requirements.txt

 2.  For **conda**: Optional but recommended, configure a virtual env for your work:  
 
  To create the environment and install all requirements:
  > conda env create -f environment.yml
  
  To load the environment
  > source activate bdraanalytics
  
  If the environment has been updated, can update the environment with:
  > conda env update --name=bdranalytics -f environment.yml

## Development on this package

### Setup details and alternative setups
This package has dependencies and requirements:
  * dependencies will be always be installed through a pip install of the module / setup.py
  * requirements are required before installing the dependencies. For example, scikit-learn requies numpy.

### Creating a package dist

To create a dist from a local checkout (when developing on this module):
> python setup.py sdist

### Running the installation script
This uses the setup.py script directly, useful for testing how the dist will be installed without creating the dist.

To just install the package and main dependencies from a local checkout (when going to use this module):
> python setup.py install

