from setuptools import setup

setup(
    name='bdranalytics',
    version='0.1',
    license='within_BDR_only',
    author='geroos',
    author_email='gerben.oostra@bigdatarepublic.nl',
    url='http://www.bigdatarepublic.nl',
    long_description="README.txt",
    packages=['bdranalytics', 'bdranalytics.images'],
    include_package_data=True,
    package_data={'bdranalytics': ['data/*.dat'],
                  'bdranalytics.images': ['bdr.gif']},
    description="Our analytics library to quickly get our data scientists up to speed",
    install_requires=[
        'sklearn>=1.18'
    ]
)