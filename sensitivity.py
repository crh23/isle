"""
This script allows to launch an ensemble of simulations for different number of risks models.
It can be run locally if no argument is passed when called from the terminal.
It can be run in the cloud if it is passed as argument the sandman2 server that will be used.
"""
import sys
import os
from typing import Dict
import time
import importlib

import numpy as np
import sandman2.api as sm

import isleconfig
import start
import setup_simulation


def rake(hostname=None, summary: callable = None):
    """
    Uses the sandman2 api to run multiple replications of multiple configurations of the simulation.
    If hostname=None, runs locally. Otherwise, make sure environment variable SANDMAN_KEY_ID and SANDMAN_KEY_SECRET
    are set.
    Args:
        hostname: The remote server to run the job on
        summary: The summary statistic (function) to apply to the results
    """
    if importlib.util.find_spec("hickle") is None:
        raise ModuleNotFoundError("hickle not found but required for saving logs")

    if hostname is None:
        print("Running ensemble locally")
    else:
        print(f"Running ensemble on {hostname}")
    """Configure the parameter sets to run"""
    default_parameters: Dict = isleconfig.simulation_parameters
    parameter_list = None

    ###################################################################################################################
    # This section should be freely modified to determine the experiment
    # The keys of parameter_sets are the prefixes to save logs under, the values are the parameters to run
    # The keys should be strings

    import SALib.util
    import SALib.sample.morris

    problem = SALib.util.read_param_file("isle_all_parameters.txt")
    param_values = SALib.sample.morris.sample(problem, N=problem["num_vars"] * 3)
    parameters = [tuple(row) for row in param_values]
    parameter_list = [
        {problem["names"][i]: row[i] for i in range(len(row))} for row in param_values
    ]
    for d in parameter_list:
        d.update(default_parameters.copy())
        d["max_time"] = 2000
    assert parameter_list[1] != parameter_list[0]

    ###################################################################################################################

    max_time = isleconfig.simulation_parameters["max_time"]

    print(f"Running {len(parameter_list)} simulations of {max_time} timesteps")

    """Sanity checks"""

    # Check that the necessary env variables are set
    if hostname is not None:
        if not ("SANDMAN_KEY_ID" in os.environ and "SANDMAN_KEY_SECRET" in os.environ):
            print("Warning: Sandman authentication not found in environment variables.")

    if hostname is not None and isleconfig.show_network:
        print("Warning: can't show network on remote server")
        isleconfig.show_network = False

    """Configuration of the ensemble"""

    """Configure the return values and corresponding file suffixes where they should be saved"""
    requested_logs = {
        "total_cash": "_cash.dat",
        "total_excess_capital": "_excess_capital.dat",
        "total_profitslosses": "_profitslosses.dat",
        "total_contracts": "_contracts.dat",
        "total_operational": "_operational.dat",
        "total_reincash": "_reincash.dat",
        "total_reinexcess_capital": "_reinexcess_capital.dat",
        "total_reinprofitslosses": "_reinprofitslosses.dat",
        "total_reincontracts": "_reincontracts.dat",
        "total_reinoperational": "_reinoperational.dat",
        "total_catbondsoperational": "_total_catbondsoperational.dat",
        "market_premium": "_premium.dat",
        "market_reinpremium": "_reinpremium.dat",
        "cumulative_bankruptcies": "_cumulative_bankruptcies.dat",
        "cumulative_market_exits": "_cumulative_market_exits.dat",
        "cumulative_unrecovered_claims": "_cumulative_unrecovered_claims.dat",
        "cumulative_claims": "_cumulative_claims.dat",
        "cumulative_bought_firms": "_cumulative_bought_firms.dat",
        "cumulative_nonregulation_firms": "_cumulative_nonregulation_firms.dat",
        "insurance_firms_cash": "_insurance_firms_cash.dat",
        "reinsurance_firms_cash": "_reinsurance_firms_cash.dat",
        "market_diffvar": "_market_diffvar.dat",
        "rc_event_schedule_initial": "_rc_event_schedule.dat",
        "rc_event_damage_initial": "_rc_event_damage.dat",
        "number_riskmodels": "_number_riskmodels.dat",
        "individual_contracts": "_insurance_contracts.dat",
        "reinsurance_contracts": "_reinsurance_contracts.dat",
        "unweighted_network_data": "_unweighted_network_data.dat",
        "network_node_labels": "_network_node_labels.dat",
        "network_edge_labels": "_network_edge_labels.dat",
        "number_of_agents": "_number_of_agents",
    }
    """Define the numpy types of the underlying data in each requested log"""
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
        "insurance_firms_cash": np.float_,
        "reinsurance_firms_cash": np.float_,
        "market_diffvar": np.float_,
        "rc_event_schedule_initial": np.int_,
        "rc_event_damage_initial": np.float_,
        "number_riskmodels": np.int_,
        "individual_contracts": np.int_,
        "reinsurance_contracts": np.int_,
        "unweighted_network_data": np.float_,
        "network_node_labels": np.float_,
        "network_edge_labels": np.float_,
        "number_of_agents": np.int_,
    }

    if isleconfig.slim_log:
        for name in [
            "insurance_firms_cash",
            "reinsurance_firms_cash",
            "individual_contracts",
            "reinsurance_contracts",
            "unweighted_network_data",
            "network_node_labels",
            "network_edge_labels",
            "number_of_agents",
        ]:
            del requested_logs[name]

    elif not isleconfig.save_network:
        for name in [
            "unweighted_network_data",
            "network_node_labels",
            "network_edge_labels",
            "number_of_agents",
        ]:
            del requested_logs[name]

    """Configure log directory and ensure that the directory exists"""
    dir_prefix = "/data/"
    directory = os.getcwd() + dir_prefix
    if not os.path.isdir(directory):
        if os.path.exists(directory.rstrip("/")):
            raise Exception(
                "./data exists as regular file. "
                "This filename is required for the logging and event schedule directory"
            )
        os.makedirs("data")

    """Setup of the simulations"""
    # Here the setup for the simulation is done.
    # Since this script is used to carry out simulations in the cloud will usually have more than 1 replication.
    # We don't set filepath=, so the full set of events and seeds will be stored in data/risk_event_schedules.islestore
    # If we wished we could replicate by setting isleconfig.replicating = True.
    setup = setup_simulation.SetupSim()
    [
        general_rc_event_schedule,
        general_rc_event_damage,
        np_seeds,
        random_seeds,
    ] = setup.obtain_ensemble(len(parameter_list))

    m = sm.operation(start.main, include_modules=True)

    # Here is assembled each job with the corresponding: simulation parameters, time events, damage events, seeds,
    # simulation state save interval (never), and list of requested logs.
    job = [
        m(
            parameter_list[x],
            general_rc_event_schedule[x],
            general_rc_event_damage[x],
            np_seeds[x],
            random_seeds[x],
            0,
            0,
            list(requested_logs.keys()),
            summary=summary,
        )
        for x in range(len(parameter_list))
    ]
    """Here the jobs are submitted"""
    print("Jobs constructed, submitting")
    with sm.Session(host=hostname, default_cb_to_stdout=True) as sess:
        print("Starting job")
        # Don't use async here, since there is only one job
        result = sess.submit(job)

    result_dict = {t: r for t, r in zip(parameters, result)}
    start.save_summary([result_dict])


def get_cumulative_bankruptcies(log):
    return log["cumulative_bankruptcies"][-1]


if __name__ == "__main__":
    host = None
    if len(sys.argv) > 1:
        # The server is passed as an argument.
        host = sys.argv[1]
    rake(host, summary=get_cumulative_bankruptcies)
    # jobs = {"ensemble1" : "23a3f4e1",
    #         "ensemble2" : "485f7221"}
    # restore_jobs(jobs, host)
