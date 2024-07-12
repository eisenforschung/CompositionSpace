# CompositionSpace
CompositionSpace is a python library for analysis of APT data.

## Installation

### Installation for developers on your local machine into a virtual environment:
```
git clone https://github.com/eisenforschung/CompositionSpace
cd CompositionSpace
git submodule sync --recursive
git submodule update --init --recursive --remote
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```
<!--
### Installation for users via [PyPI](https://pypi.org/)

CompositionSpace can be installed using:

```
pip install compositionspace
```-->

<!--
### Installation for users via [Conda](https://anaconda.org/)
It is recommended to install and use `compositionspace` within a conda environment. To see how you can install conda see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Once a conda distribution is available, the following steps will help set up an environment to use `compositionspace`. First step is to clone the repository.

```
git clone https://github.com/eisenforschung/CompositionSpace.git
```

After cloning, an environment can be created from the included file-

```
cd CompositionSpace
conda env create -f environment.yml
```

Activate the environment,

```
conda activate compspace
```

then, install `compositionspace` using,

```
python setup.py install
```

The environment is now set up to run compositionspace.
-->

## Getting started
Navigate to tests. Spin up a jupyter notebook and run `FullWorkflow.ipynb`.

[The usa_denton_smith dataset is available here](https://zenodo.org/records/7986279/files/usa_denton_smith_apav_si.zip?download=1)
[Further atom probe datasets for testing are available here](https://dx.doi.org/10.25833/3ge0-y420)

<!--
## Documentation

Documentation is available [here](https://compositionspace.readthedocs.io/en/latest/).
-->
