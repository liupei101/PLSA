from setuptools import setup, find_packages

with open('README.txt') as file:
    long_description = file.read()

setup(name='PLSA',
    version='0.2',
    description='A Python Library of Statistical Analyze for using in my private workflow.',
    keywords = "survival analysis machine learning workflow",
    url='https://github.com/liupei101/PLSA',
    author='Pei Liu',
    author_email='18200144374@163.com',
    license='MIT',
    long_description = long_description,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python 2.7",
        "Topic :: Scientific/Engineering",
    ],
    packages = find_packages(),
    install_requires=[
        'lifelines>=0.9.2',
        'pandas>=0.18',
        'scikit-learn>=0.19.0',
        'pyper',
        'seaborn',
        'sklearn2pmml',
        'xgboost',
        'statsmodels',
    ],
    include_package_data=True,
)
