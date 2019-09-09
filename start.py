# import common packages

import argparse
import hashlib
import numpy as np
import os
import pickle
import zlib
import random
from typing import MutableMapping, MutableSequence, List, Tuple

import calibrationscore
import insurancesimulation
import listify
import calibration_statistic

# import config file and apply configuration
import isleconfig
import logger

simulation_parameters = isleconfig.simulation_parameters
filepath = None
overwrite = False
override_no_riskmodels = False


"""Creates data file for logs if does not exist"""
if not os.path.isdir("data"):
    if os.path.exists("data"):
        raise FileExistsError(
            "./data exists as regular file. This filename is required for the logging directory"
        )
    os.makedirs("data")


def cumulative_bankruptcies(log):
    return log["cumulative_bankruptcies"][-1]


# main function
def main(
    sim_params: MutableMapping,
    rc_event_schedule: MutableSequence[MutableSequence[int]],
    rc_event_damage: MutableSequence[MutableSequence[float]],
    np_seed: int,
    random_seed: int,
    save_iteration: int,
    replic_id: int,
    requested_logs: MutableSequence = None,
    resume: bool = False,
    summary: callable = None,
) -> Tuple[bytes, dict]:
    if not resume:
        np.random.seed(np_seed)
        random.seed(random_seed)

        sim_params["simulation"] = simulation = insurancesimulation.InsuranceSimulation(
            override_no_riskmodels,
            replic_id,
            sim_params,
            rc_event_schedule,
            rc_event_damage,
        )
        time = 0
    else:
        d = load_simulation()
        np.random.set_state(d["np_seed"])
        random.setstate(d["random_seed"])
        time = d["time"]
        simulation = d["simulation"]
        sim_params = d["simulation_parameters"]
        for key in d["isleconfig"]:
            isleconfig.__dict__[key] = d["isleconfig"][key]
        isleconfig.simulation_parameters = sim_params
    for t in range(time, sim_params["max_time"]):
        # Main time iteration loop
        simulation.iterate(t)

        # log data
        simulation.save_data()

        # Don't save at t=0 or if the simulation has just finished
        if (
            save_iteration > 0
            and t % save_iteration == 0
            and 0 < t < sim_params["max_time"]
        ):
            # Need to use t+1 as resume will start at time saved
            save_simulation(t + 1, simulation, sim_params, exit_now=False)

    log = simulation.obtain_log(requested_logs)
    if summary is not None:
        return summary(log)
    else:
        # We compute metadata about the return data that isn't compressed, so a skeleton data structure can be
        # constructed before decompression
        found_shapes = {name: np.shape(log[name]) for name in log}

        # We compress the return value for the sake of minimising data transfer over the network and RAM usage
        return (zlib.compress(pickle.dumps(log)), found_shapes)


def save_simulation(
    t: int,
    sim: insurancesimulation.InsuranceSimulation,
    sim_param: MutableMapping,
    exit_now: bool = False,
) -> None:
    d = {
        "np_seed": np.random.get_state(),
        "random_seed": random.getstate(),
        "time": t,
        "simulation": sim,
        "simulation_parameters": sim_param,
        "isleconfig": {},
    }
    for key in isleconfig.__dict__:
        if not key.startswith("__"):
            d["isleconfig"][key] = isleconfig.__dict__[key]

    with open("data/simulation_save.pkl", "bw") as wfile:
        pickle.dump(d, wfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open("data/simulation_save.pkl", "br") as rfile:
        file_contents = rfile.read()
    print(
        "\nSaved simulation with hash:",
        hashlib.sha512(str(file_contents).encode()).hexdigest(),
    )

    if exit_now:
        exit(0)


def load_simulation() -> dict:
    # TODO: Fix! This doesn't work, the retrieved file is different to the saved one.
    with open("data/simulation_save.pkl", "br") as rfile:
        print(
            "\nLoading simulation with hash:",
            hashlib.sha512(str(rfile.read()).encode()).hexdigest(),
        )
        rfile.seek(0)
        file_contents = pickle.load(rfile)
    return file_contents


def save_results(results_list: list, prefix: str):
    """Saves the results of a simulation run to disk. results_list is a list of tuples, where each tuple consists of
     a compressed, pickled dict and a dict of the shapes of the data in the compressed dict (metadata)"""
    # We use the first metadata to infer basic shape information
    current_shapes = results_list[0][1]
    replications = len(results_list)

    types = {
        "total_cash": np.float_,
        "total_excess_capital": np.float_,
        "total_profitslosses": np.float_,
        "total_contracts": np.int_,
        "total_operational": np.int_,
        "total_reincash": np.float_,
        "total_reinexcess_capital": np.float_,
        "total_reinprofitslosses": np.float_,
        "total_reincontracts": np.int_,
        "total_reinoperational": np.int_,
        "total_catbondsoperational": np.int_,
        "market_premium": np.float_,
        "market_reinpremium": np.float_,
        "cumulative_bankruptcies": np.int_,
        "cumulative_market_exits": np.int_,
        "cumulative_unrecovered_claims": np.float_,
        "cumulative_claims": np.float_,
        "cumulative_bought_firms": np.int_,
        "cumulative_nonregulation_firms": np.int_,
        "market_diffvar": np.float_,
        # Would store these two as an array of lists, but hdf5 can't do that
        "rc_event_schedule_initial": np.object,
        "rc_event_damage_initial": np.object,
        "number_riskmodels": np.int_,
        "unweighted_network_data": np.float_,
        "network_node_labels": np.float_,
        "network_edge_labels": np.float_,
        "number_of_agents": np.int_,
        "insurance_cumulative_dividends": np.float_,
        "reinsurance_cumulative_dividends": np.float_,
        # These are the big ones, so we need to pay attention to data types
        "insurance_firms_cash": np.float32,
        "reinsurance_firms_cash": np.float32,
        "insurance_contracts": np.uint16,
        "reinsurance_contracts": np.uint16,
    }
    # bad_logs are the logs that don't have a consistent size between replications
    bad_logs = [
        "rc_event_schedule_initial",
        "rc_event_damage_initial",
        "insurance_contracts",
        "insurance_firms_cash",
        "reinsurance_contracts",
        "reinsurance_firms_cash",
    ]
    event_info_names = ["rc_event_schedule_initial", "rc_event_damage_initial"]

    logs_found = current_shapes.keys()
    for name in logs_found:
        if name not in types:
            print(f"Warning: type of log {name} not known, assuming float")
            types[name] = np.float_
    shapes = {}
    for name in logs_found:
        if name not in bad_logs:
            # These are mostly standard 1-d timeseries, but may also include stuff like no_riskmodels
            shapes[name] = (replications,) + current_shapes[name]
        else:
            # We could probably do this for all of the data, but this is fine for now.
            # These are sets of timeseries: the sets have variable size (also the event schedules)
            # We use the uncompressed metadata
            found_shapes = [result[1][name] for result in results_list]
            # This only works because the shapes only vary in one dimension (tuple comparison is lexicographic)
            shapes[name] = (replications,) + max(found_shapes)

    # Make a skeleton data structure so we only need to have one uncompressed log in memory at a time
    results_dict = {
        name: np.zeros(shape=shapes[name], dtype=types[name])
        for name in current_shapes.keys()
    }
    # results_dict is a dictionary of numpy arrays, should be efficient to store.
    # The event schedules/damages are of differing lengths. Could pad them with NaNs, but probably
    # would be more trouble than it's worth

    for i, result_tuple in enumerate(results_list):
        result = pickle.loads(zlib.decompress(result_tuple[0]))
        for name in results_dict:
            if (name not in event_info_names) and hasattr(result[name], "__len__"):
                arr = np.asarray(result[name])
                shape_slice = tuple([slice(i) for i in arr.shape])
                results_dict[name][i][shape_slice] = result[name]
            else:
                results_dict[name][i] = result[name]

    # Need to do a little pre-processing
    for key in list(results_dict.keys()):
        if not isinstance(results_dict[key], np.ndarray):
            raise ValueError(f"Results_dict[{key}] is not an array")
        if results_dict[key].size == 0:
            del results_dict[key]
            continue
        if results_dict[key].dtype == np.object:
            results_dict[key] = results_dict[key].tolist()
    data = results_dict
    # data = (True, (results_dict, event_info))
    # We store everything in one file(!)

    filename = "data/" + prefix + "_full_logs.hdf"

    if os.path.exists(filename):
        # Don't want to blindly overwrite, so make backups
        import time

        backupfilename = filename + "." + time.strftime("%Y-%m-%dT%H%M%S")
        os.rename(filename, backupfilename)
    # data is a tuple, first element indicating whether the logs are slim, second element being the data
    # TODO: Make everything else work with this new format
    # Import here so sandman never tries to import
    import hickle

    hickle.dump(data, filename, compression="gzip")


def save_summary(summary_values: List[dict]):
    filename = "data/summary_statistics.hdf"
    if os.path.exists(filename):
        # Don't want to blindly overwrite, so make backups
        import time

        backupfilename = filename + "." + time.strftime("%Y-%m-%dT%H%M%S")
        os.rename(filename, backupfilename)
    import hickle

    hickle.dump(summary_values, filename, compression="gzip")


# main entry point
if __name__ == "__main__":

    """ use argparse to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Model the Insurance sector")
    parser.add_argument(
        "-f",
        "--file",
        action="store",
        help="the file to store the initial randomness in. Will be stored in ./data and appended with .islestore "
        "(if it is not already). The default filepath is ./data/risk_event_schedules.islestore, which will be "
        "overwritten event if --overwrite is not passed!",
    )
    parser.add_argument(
        "-r",
        "--replicating",
        action="store_true",
        help="if this is a simulation run designed to replicate another, override the config file parameter. "
        "You probably want to specify the --file to read from.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="allows overwriting of the file specified by -f",
    )
    parser.add_argument(
        "-p", "--showprogress", action="store_true", help="show timesteps"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="more detailed output"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the simulation from a previous save in ./data/simulation_save.pkl. "
        "All other arguments will be ignored",
    )
    parser.add_argument(
        "--riskmodels",
        type=int,
        choices=[1, 2, 3, 4],
        help="allow overriding the number of riskmodels from standard config (with 1 or other numbers)."
        " Overrides --oneriskmodel",
    )
    parser.add_argument(
        "--randomseed", type=float, help="allow setting of numpy random seed"
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="force foreground runs even if replication ID is given (which defaults to background runs)",
    )
    parser.add_argument(
        "--shownetwork",
        action="store_true",
        help="show reinsurance relations as network",
    )
    parser.add_argument(
        "--save_iterations",
        type=int,
        help="number of iterations to iterate before saving world state",
    )
    args = parser.parse_args()

    if args.riskmodels:
        override_no_riskmodels = args.riskmodels
    if args.file:
        filepath = args.file
    if args.overwrite:
        overwrite = True
    if args.replicating:
        isleconfig.replicating = True
    if args.randomseed:
        randomseed = args.randomseed
        seed = int(randomseed)
    else:
        np.random.seed()
        seed = np.random.randint(0, 2 ** 31 - 1)
    if args.foreground:
        isleconfig.force_foreground = True
    if args.shownetwork:
        isleconfig.show_network = True
    if args.showprogress:
        isleconfig.showprogress = True
    if args.verbose:
        isleconfig.verbose = True
    if args.save_iterations:
        save_iter = args.save_iterations
    else:
        # Disable saving unless save_iter is given. It doesn't work anyway # TODO
        save_iter = isleconfig.simulation_parameters["max_time"] + 2

    if not args.resume:
        from setup_simulation import SetupSim

        setup = SetupSim()  # Here the setup for the simulation is done.

        # Only one ensemble. This part will only be run locally (laptop).
        [
            general_rc_event_schedule,
            general_rc_event_damage,
            np_seeds,
            random_seeds,
        ] = setup.obtain_ensemble(1, filepath, overwrite)
    else:
        # We are resuming, so all of the necessary setup will be loaded from a file
        general_rc_event_schedule = (
            general_rc_event_damage
        ) = np_seeds = random_seeds = [None]

    summary = calibration_statistic.calculate_single
    # Run the main program
    comp_result = main(
        simulation_parameters,
        general_rc_event_schedule[0],
        general_rc_event_damage[0],
        np_seeds[0],
        random_seeds[0],
        save_iter,
        replic_id=1,
        resume=args.resume,
        summary=summary,
    )
    if summary is None:
        save_results([comp_result], "single")

        decomp_result = pickle.loads(zlib.decompress(comp_result[0]))
        L = logger.Logger()
        L.restore_logger_object(decomp_result)
        if isleconfig.save_network:
            L.save_network_data(ensemble=False)

        """ Obtain calibration score """
        CS = calibrationscore.CalibrationScore(L)
        score = CS.test_all()
    else:
        print("\nSummary output:")
        print(comp_result)
