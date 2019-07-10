[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d46ac6670e8a4016a382434445668d70)](https://www.codacy.com/app/herculesl/isle?utm_source=github.com&utm_medium=referral&utm_content=EconomicSL/isle&utm_campaign=badger)

# isle

# Installation of dependencies

If the dependencies haven't been installed, run this command in a terminal

```
$ pip install -r requirements.txt
```

# Usage

## Simulation 

Execute a single simulation run with the command:

```
$ python3 start.py
```

## Simulation with additional options

The ```start.py``` script accepts a number of options. 

```
usage: start.py [--oneriskmodel] [--riskmodels {1,2,3,4}] [--replicid REPLICID]
                [--replicating] [--randomseed RANDOMSEED] [--foreground]
                [--shownetwork] [-p] [-v] [--save_iterations]
```

See the help for more details

```
python3 start.py --help
```

## Ensemble simulations

The bash scripts ```starter_*.sh``` can be used to run ensembles of a large number of simulations for settings with 1-4 different risk models. ```starter_two.sh``` is set up to generate random seeds and risk event schedules that are - for consistency and comparability - also used by the other scripts (i.e. ```starter_two.sh``` needs to be run first).

```
bash starter_two.sh
bash starter_one.sh
bash starter_four.sh
bash starter_three.sh
```

## Plotting

####Single runs
Use the script ```plotter.py``` to plot insurer and reinsurer data, or run  
```visualisation.py [--single]``` from the command line to plot this and visualise the run.

####Ensemble runs
Use  ```metaplotter_pl_timescale.py```,  ```metaplotter_pl_timescale_additional_measures.py```, 
or ```visualisation.py [--comparison]``` to visualize ensemble runs.

