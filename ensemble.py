"""
This script allows to launch an ensemble of simulations for different number of risks models.
It can be run locally if no argument is passed when called from the terminal.
It can be run in the cloud if it is passed as argument the sandman2 server that will be used.
"""
import sys
import os
from typing import Dict
import time

import numpy as np
import sandman2.api as sm

import isleconfig
import start
import setup_simulation


def rake(hostname=None, replications=9):
    """
    Uses the sandman2 api to run multiple replications of multiple configurations of the simulation.
    If hostname=None, runs locally. Otherwise, make sure environment variable SANDMAN_KEY_ID and SANDMAN_KEY_SECRET
    are set.
    Args:
        hostname: The remote server to run the job on
        replications: The number of replications of each parameter set to run
    """
    if hostname is None:
        print("Running ensemble locally")
    else:
        print(f"Running ensemble on {hostname}")
    """Configure the parameter sets to run"""
    default_parameters: Dict = isleconfig.simulation_parameters
    parameter_sets: Dict[str:Dict] = {}

    ###################################################################################################################
    # This section should be freely modified to determine the experiment
    # The keys of parameter_sets are the prefixes to save logs under, the values are the parameters to run
    # The keys should be strings

    for number_riskmodels in [1, 3]:
        # default_parameters is mutable, so should be copied
        new_parameters = default_parameters.copy()
        new_parameters["no_riskmodels"] = number_riskmodels
        parameter_sets["ensemble" + str(number_riskmodels)] = new_parameters

    ###################################################################################################################

    print(
        f"Running {len(parameter_sets)} simulations of {replications} "
        f"replications of {default_parameters['max_time']} timesteps"
    )
    for name in parameter_sets:
        assert isinstance(name, str)

    """Sanity checks"""

    # Check that the necessary env variables are set
    if hostname is not None:
        if not ("SANDMAN_KEY_ID" in os.environ and "SANDMAN_KEY_SECRET" in os.environ):
            print("Warning: Sandman authentication not found in environment variables.")

    max_time = isleconfig.simulation_parameters["max_time"]

    if not isleconfig.slim_log:
        # We can estimate the log size per experiment in GB (max_time is squared as number of insurance firms also
        # increases with time and per-firm logs are dominating in the limit). The 6 is empirical
        # TODO: Is this even vaguely correct? Who knows!
        estimated_log_size = max_time ** 2 * replications * 6 / (1000 ** 3)
        if estimated_log_size > 1:
            print(
                "Uncompressed log size estimated to be above 1GB - consider using slim logs"
            )

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

    """Clear old dict saving files (*_history_logs.dat)"""
    for prefix in parameter_sets.keys():
        filename = os.getcwd() + dir_prefix + "full_" + prefix + "_history_logs.dat"
        if os.path.exists(filename):
            os.remove(filename)

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
    ] = setup.obtain_ensemble(replications)

    # never save simulation state in ensemble runs (resuming is impossible anyway)
    save_iter = 0

    m = sm.operation(start.main, include_modules=True)

    jobs = {}
    position_maps = {}
    for prefix in parameter_sets:
        # In this loop the parameters, schedules and random seeds for every run are prepared. Different risk models will
        # be run with the same schedule, damage size and random seed for a fair comparison.

        simulation_parameters = parameter_sets[prefix]

        # Here is assembled each job with the corresponding: simulation parameters, time events, damage events, seeds,
        # simulation state save interval (never), and list of requested logs.
        job = [
            m(
                simulation_parameters,
                general_rc_event_schedule[x],
                general_rc_event_damage[x],
                np_seeds[x],
                random_seeds[x],
                save_iter,
                0,
                list(requested_logs.keys()),
            )
            for x in range(replications)
        ]
        jobs[prefix] = job
        position_maps[prefix] = {o.id: p for p, o in enumerate(job)}

    """Here the jobs are submitted"""
    print("Jobs constructed, submitting")
    with sm.Session(host=hostname, default_cb_to_stdout=True) as sess:
        # TODO: Allow for resuming a detatched run with task = sess.get(job_id)
        tasks = {}
        for prefix, job in jobs.items():
            # If there are 4 parameter sets jobs will be a dict with 4 elements.

            """Run simulation and obtain result"""
            task = sess.submit_async(job)
            print(f"Started job, prefix {prefix}, given ID {task.id}")
            tasks[prefix] = task

        print("Now waiting for jobs to complete\033[5m...\033[0m")
        wait_for_tasks(tasks, replications, position_maps)

    print("Recieved all results and written all files, all finished.")


def wait_for_tasks(tasks: dict, replications: int, position_maps: dict):
    """tasks is a dict mapping prefixes to job objects
    position_maps is a dict of dicts: maps prefixes to dicts maping ids to positions
    """
    # Need to do it this way as dictionary can't change size during iteration
    completed_tasks = []
    while len(tasks) > 0:
        for prefix in completed_tasks:
            del tasks[prefix]
        completed_tasks = []

        time.sleep(0.5)
        for prefix, task in tasks.items():
            if task.is_done():
                print(f"Finsihed job, prefix {prefix}, with ID {task.id}")
                results_iterator = task.iterresults()  # Could just use .results()?
                completed_tasks.append(prefix)

                results_list = [None for _ in range(replications)]
                for output_id, result in results_iterator:
                    position = position_maps[prefix][output_id]
                    results_list[position] = result
                # Note that the results are still compressed and pickled
                print(
                    f"Obtained compressed results for job {task.id}, writing to "
                    f"file {'data/' + prefix + '_full_logs.hdf'}"
                )
                start.save_results(results_list, prefix)
                print(f"Finished writing results for job {task.id}")


# TODO: Currently broken due to a sandman bug
def restore_jobs(jobs, hostname):
    """jobs is a dict mapping prefixes to job ids"""
    # Can't restore jobs on a local scheduler
    assert hostname is not None
    with sm.Session(host=hostname, default_cb_to_stdout=True) as sess:
        tasks = {prefix: sess.get(jobs[prefix]) for prefix in jobs}
        # Might need to store the position maps - can't test what can be extracted until sandman is fixed
        # position_maps = None
        # replications = list(tasks.values())[0].f


if __name__ == "__main__":
    host = None
    if len(sys.argv) > 1:
        # The server is passed as an argument.
        host = sys.argv[1]
    rake(host)
    # jobs = {"ensemble1" : "23a3f4e1",
    #         "ensemble2" : "485f7221"}
    # restore_jobs(jobs, host)
