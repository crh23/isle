[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d46ac6670e8a4016a382434445668d70)](https://www.codacy.com/app/herculesl/isle?utm_source=github.com&utm_medium=referral&utm_content=EconomicSL/isle&utm_campaign=badger)

# isle

# Installation of dependencies

If the dependencies haven't been installed, run this command in a terminal

```
$ pip install -r requirements.txt
```

# Usage

Isle requires that `./data` does not exist as a file, and may overwrite 
particular file names in `./data/`.

## Simulation 

Execute a single simulation run with the command:

```
$ python3 start.py
```

## Simulation with additional options

The ```start.py``` script accepts a number of options. 

```
usage: start.py [-h] [-f FILE] [-r] [-o] [-p] [-v] [--resume] [--oneriskmodel]
                [--riskmodels {1,2,3,4}] [--randomseed RANDOMSEED]
                [--foreground] [--shownetwork]
                [--save_iterations SAVE_ITERATIONS]
```

See the help for more details

```
$ python3 start.py --help
```

## Ensemble simulations

The bash scripts ```starter_*.sh``` can be used to run ensembles of a large number of simulations for settings with 1-4 different risk models. ```starter_two.sh``` is set up to generate random seeds and risk event schedules that are - for consistency and comparability - also used by the other scripts (i.e. ```starter_two.sh``` needs to be run first).

```
bash starter_two.sh
bash starter_one.sh
bash starter_four.sh
bash starter_three.sh
```

## Visualisation

#### Single runs
Use the script ```visualisation.py [--single]``` from the command line to plot data from a single run. It also takes the 
arguments ```[--pie] [--timeseries]``` for which data representation is wanted. The argument ```[--config_compare_filex ]``` 
where ```x``` can be 1,2 or 3 is used for comparing two sets of data (singular or with replications) with different conditions.

If the necessary data has been saved a network animation can also be created by running ```visualization_network.py``` 
which takes the arguments ```[--save] [--number_iterations]``` if you want the animation to be saved as an mp4, and how
many time iterations you want in the animation.

#### Ensemble runs
Ensemble runs can be plotted if the correct data is available using ``visualisation.py`` which has a number of arguments.

```
visualiation.py [--timeseries_comparison] [--firmdistribution] 
                [--bankruptcydistribution] [--compare_riskmodels]
```

See help for more information.

# Contributing

## Code style

[PEP 8](https://www.python.org/dev/peps/pep-0008/) styling should be used where possible. 
The Python code formatter [black](https://github.com/python/black) is a good way
to automatically fix style problems - install it with `$ pip install black` and
then run it with, say, `black *.py`. Additionally, it is good to run flake8 over your code.
