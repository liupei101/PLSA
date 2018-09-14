from setuptools import setup, find_packages

with open('README.txt') as file:
    long_description = file.read()

setup(name='PLSA',
    version='0.1',
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
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    packages = find_packages(),
    install_requires=[
        'scikit-learn>=0.19.0',
        'lifelines>=0.9.2',
    ],
    include_package_data=True,
)