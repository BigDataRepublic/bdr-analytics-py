from setuptools import setup

setup(
    name='bdranalytics',
    version='0.2',
    license='Apache License 2.0',
    author='bigdatarepublic',
    author_email='info@bigdatarepublic.nl',
    url='http://www.bigdatarepublic.nl',
    long_description="README.md",
    packages=['bdranalytics',
              'bdranalytics.images',
              'bdranalytics.keras',
              'bdranalytics.pdlearn',
              'bdranalytics.plot',
              'bdranalytics.sklearn'],
    include_package_data=True,
    package_data={'bdranalytics': ['data/*.dat'],
                  'bdranalytics.images': ['bdr.gif']},
    description="Making data science workflows easier.",
    install_requires=[
        "NumPy>=1.6.1",
        "SciPy>=0.9",
        "scikit-learn>=0.18",
        "keras",
        "pandas",
        "matplotlib",
    ],
    test_suite='nose.collector',
    tests_require=['nose',
                   "pytest",
                   "pytest-runner"]
)
