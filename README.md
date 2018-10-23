# PLSA

![Documentation Status](https://readthedocs.org/projects/plsa/badge/?version=latest) ![Python Version](https://img.shields.io/badge/Python-2.7-yellow.svg)

A Python Library of Statistical Analyze for using in my private workflow.

# Introduction

This project is a Python Library of Statistical Analyze(PLSA). 

The library collects files generated in my routinue, and motivation of it is mainly reusing. Have to mention that this package would be suitable for people working in field of medical statistical analysis.

This library will be updated and released as a standard Python Pypi Packages.

Functions in the package is summarized below respectively:

- data
    - processing: process data used in survival analyze.
- qcal
    - func: integrate functions of other mudules, mainly for calling freely.
- surv
    - cutoff: get optimal cutoffs according different criterion in survival analyze.
    - utils: include general function in survival analyze.
- utils
    - cutoff: get optimal cutoffs according different criterion in general case.
    - metrics: evaluate models by some metrics.
    - test: include methods of hypothetical test.
    - write: save and output formatted PMML file converted from sklearn model.
- vision
    - calibration: visualize calibration curve.
    - roc: visualize ROC curve.
    - survrisk: visualize survial function of different risk-groups.
    - lib: include general function in plotting figure.

# Read Docs

[PLSA](http://plsa.readthedocs.io/)

# TODO list

None

# News

- Version 0.2 is released.
- Support for ploting forest of coefficient in CPH model
