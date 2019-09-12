import sandman2.api as sm
import pyabc
import pyabc.sampler
from typing import Callable, Iterable, TypeVar, List
from functools import partial
import os
import isleconfig
import setup_simulation
import start
import calibration_statistic
import scipy.spatial
import scipy.stats
import numpy as np

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def sm_map(
    func: Callable[[T1], T2], iter_: Iterable[T1], hostname: str = None
) -> List[T2]:
    """Implements a map-like interface using sandman. Should be obtained using get_sm_map to set hostname"""
    op = sm.operation(func, include_modules=True)
    outputs = [op(arg) for arg in iter_]
    with sm.Session(host=hostname) as sess:
        result = sess.submit(outputs)
    return result


def get_sm_map(hostname: str) -> Callable[[Callable[[T1], T2], Iterable[T1]], List[T2]]:
    """Returns sm_map with hostname parameter pre-filled, so it can be used as a drop-in repacement for
    map (with a single iterable)"""
    return partial(sm_map, hostname=hostname)


def model(parameters: dict) -> dict:
    """Runs the model with random randomness
    Args:
        parameters (dict): the parameters of this model run. Override those in isleconfig
    Returns:
        the result as a dictionary
    """
    sim_params = isleconfig.simulation_parameters.copy()
    sim_params.update(parameters)
    setup = setup_simulation.SetupSim()
    [event_schedule, event_damage, np_seeds, random_seeds] = setup.obtain_ensemble(
        1, overwrite=True
    )
    result = start.main(
        sim_params=sim_params,
        rc_event_schedule=event_schedule[0],
        rc_event_damage=event_damage[0],
        np_seed=np_seeds[0],
        random_seed=random_seeds[0],
        save_iteration=0,
        replic_id=0,
        requested_logs=None,
        resume=False,
        summary=calibration_statistic.calculate_single,
    )
    return result


def single_prior(lower_bound, upper_bound, shape):
    if shape not in ("linear", "logarithmic"):
        print(f"Warning: shape {shape} not recognised, assuming linear")
        shape = "linear"
    lower_bound, upper_bound = float(lower_bound), float(upper_bound)
    if shape == "linear":
        return scipy.stats.uniform(lower_bound, upper_bound - lower_bound)
    elif shape == "logarithmic":
        return scipy.stats.reciprocal(lower_bound, upper_bound)


def get_prior():
    params = {}
    param_file_path = os.path.join(os.getcwd(), "isle_calibration_parameters.txt")
    with open(param_file_path, "r") as rfile:
        for line in rfile:
            if line.startswith("#") or line == "\n":
                continue
            parts = line.strip("\n").replace(" ", "").split(",")
            if len(parts) < 4:
                continue
            params[parts[0]] = single_prior(parts[1], parts[2], parts[3])
    return params


def calibrate(observed: dict, hostname: str = None):
    """Calibrates. observed is a dictionary with keys as in calibration_statistic.statistics containing the real data"""
    db_path = "sqlite:///" + os.path.join(os.getcwd(), "data", "calibration.db")
    if hostname is not None:
        # If we're given a hostname, use the above sandman mapping wrapper
        sampler = pyabc.sampler.MappingSampler(
            map_=get_sm_map(hostname), mapper_pickles=True
        )
    else:
        # Otherwise, run locally with the normal sampler
        sampler = pyabc.sampler.MulticoreEvalParallelSampler()
    # Adaptive distance based on Prangle (2017) (also acceptor below)
    dist = pyabc.distance.AdaptivePNormDistance(p=2, adaptive=True)
    prior = pyabc.Distribution(**get_prior())
    pop_size = pyabc.populationstrategy.AdaptivePopulationSize(
        start_nr_particles=32, max_population_size=256, min_population_size=4
    )

    abc = pyabc.ABCSMC(
        model,
        parameter_priors=prior,
        distance_function=dist,
        population_size=pop_size,
        sampler=sampler,
        acceptor=pyabc.accept_use_complete_history,
    )

    run_id = abc.new(db=db_path, observed_sum_stat=observed)
    print(f"Run ID is {run_id}")
    history = abc.run(max_nr_populations=10)
    df, w = history.get_distribution()
    results = {}
    for param in df.columns.values:
        # Calculate the posterior mean of each parameter
        results[param] = np.dot(list(df[param]), list(w))

    print("Done! The results are:")
    print(results)


if __name__ == "__main__":
    import sys

    host = None
    if len(sys.argv) > 1:
        # The server is passed as an argument.
        host = sys.argv[1]
    calibrate(observed=calibration_statistic.observed, hostname=host)
