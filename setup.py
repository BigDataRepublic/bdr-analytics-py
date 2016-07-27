from setuptools import setup

setup(
    name='bdranalytics',
    version='0.1',
    license='Apache License 2.0',
    author='geroos',
    author_email='gerben.oostra@bigdatarepublic.nl',
    url='http://www.bigdatarepublic.nl',
    long_description="README.md",
    packages=['bdranalytics',
              'bdranalytics.images',
              'bdranalytics.model_selection',
              'bdranalytics.pipeline'],
    include_package_data=True,
    package_data={'bdranalytics': ['data/*.dat'],
                  'bdranalytics.images': ['bdr.gif']},
    description="Our analytics library to quickly get our data scientists up to speed",
    install_requires=[
        'sklearn>=1.18'
    ]
)
