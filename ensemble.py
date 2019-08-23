"""
This script allows to launch an ensemble of simulations for different number of risks models.
It can be run locally if no argument is passed when called from the terminal.
It can be run in the cloud if it is passed as argument the sandman2 server that will be used.
"""
import sys
import os
from typing import Dict
import time

import sandman2.api as sm

import isleconfig
import listify
import logger
import start
import setup_simulation


def rake(hostname=None, replications=50):
    """
    Uses the sandman2 api to run multiple replications of multiple configurations of the simulation.
    If hostname=None, runs locally. Otherwise, make sure environment variable SANDMAN_KEY_ID and SANDMAN_KEY_SECRET
    are set.
    Args:
        hostname: The remote server to run the job on
        replications: The number of replications of each parameter set to run
    """

    """Configure the parameter sets to run"""
    default_parameters: Dict = isleconfig.simulation_parameters
    parameter_sets: Dict[str:Dict] = {}

    ###################################################################################################################
    # This section should be freely modified to determine the experiment
    # The keys of parameter_sets are the prefixes to save logs under, the values are the parameters to run
    # The keys should be strings

    for prefix in range(1, 4):
        # default_parameters is mutable, so should be copied
        new_parameters = default_parameters.copy()
        new_parameters["no_riskmodels"] = prefix
        parameter_sets["ensemble" + str(prefix)] = new_parameters

    ###################################################################################################################

    for name in parameter_sets:
        assert isinstance(name, str)

    """Sanity checks"""

    # Check that the necessary env variables are set
    if hostname is not None:
        if not ("SANDMAN_KEY_ID" in os.environ and "SANDMAN_KEY_SECRET" in os.environ):
            print("Warning: Sandman authentication not found in environment variables.")

    # Don't want to use fat logs with lots of time steps/replications
    if (
        replications * isleconfig.simulation_parameters["max_time"] > 5000
        and not isleconfig.slim_log
    ):
        print(
            f"Warning: not using slim logs with {replications} replications and "
            f"{isleconfig.simulation_parameters['max_time']} timesteps, logs may be very large"
        )

    if hostname is not None and isleconfig.show_network:
        print("Warning: can't show network on remote server")

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

    """Here the jobs are submitted"""

    with sm.Session(host=hostname, default_cb_to_stdout=True) as sess:
        tasks = {}
        for prefix, job in jobs.items():
            # If there are 4 parameter sets jobs will be a dict with 4 elements.

            """Run simulation and obtain result"""
            task = sess.submit_async(job)
            print(f"Started job, prefix {prefix}, given ID {task.id}")
            tasks[prefix] = task

        print("Now waiting for jobs to complete\033[5m...\033[0m")
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
                    result = task.results
                    completed_tasks.append(prefix)
                    delistified_result = [
                        listify.delistify(list(res)) for res in result
                    ]

                    """These are the files created to collect the results"""
                    wfiles_dict = {}

                    logfile_dict = {}

                    for name in requested_logs.keys():
                        if "rc_event" in name or "number_riskmodels" in name:
                            logfile_dict[name] = (
                                os.getcwd()
                                + dir_prefix
                                + "check_"
                                + prefix
                                + requested_logs[name]
                            )
                        elif "firms_cash" in name:
                            logfile_dict[name] = (
                                os.getcwd()
                                + dir_prefix
                                + "record_"
                                + prefix
                                + requested_logs[name]
                            )
                        else:
                            logfile_dict[name] = (
                                os.getcwd()
                                + dir_prefix
                                + "data_"
                                + prefix
                                + requested_logs[name]
                            )

                    # with ... as would be awkward here, so use try ... finally
                    try:
                        for name in logfile_dict:
                            wfiles_dict[name] = open(logfile_dict[name], "w")

                        """Recreate logger object locally and save logs"""

                        """Create local object"""
                        log = logger.Logger()

                        for replic in range(len(job)):
                            """Populate logger object with logs obtained from remote simulation run"""
                            log.restore_logger_object(list(result[replic]))

                            """Save logs as dict (to <prefix>_history_logs.dat)"""
                            log.save_log(True, prefix=prefix)

                            """Save network data"""
                            if isleconfig.save_network:
                                log.save_network_data(ensemble=True, prefix=prefix)

                            """Save logs as individual files"""
                            for name in logfile_dict:
                                # Append the current replication data to the file
                                wfiles_dict[name].write(
                                    str(delistified_result[replic][name]) + "\n"
                                )

                    finally:
                        """Once the data is stored in disk the files are closed"""
                        for name in logfile_dict:
                            if name in wfiles_dict:
                                wfiles_dict[name].close()
                                del wfiles_dict[name]
                    print(f"Finished writing files for prefix {prefix}")


if __name__ == "__main__":
    host = None
    if len(sys.argv) > 1:
        # The server is passed as an argument.
        host = sys.argv[1]
    rake(host)
