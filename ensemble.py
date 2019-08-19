# This script allows to launch an ensemble of simulations for different number of risks models.
# It can be run locally if no argument is passed when called from the terminal.
# It can be run in the cloud if it is passed as argument the server that will be used.
import sys

import copy
import os

# noinspection PyUnresolvedReferences
from sandman2.api import operation, Session

import isleconfig
import listify
import logger
import start
from setup_simulation import SetupSim


@operation
def agg(*outputs):
    # do nothing
    return outputs


def rake(hostname):
    jobs = []

    """Configuration of the ensemble"""

    # Number of replications to be carried out for each configuration. Usually one risk model, two risk models,
    # three risk models, four risk models.
    replications = 70

    model = start.main

    m = operation(model, include_modules=True)

    riskmodels = [1, 2, 3, 4]  # The number of risk models that will be used.

    parameters = isleconfig.simulation_parameters

    nums = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

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
            "reinsurance_contracts" "unweighted_network_data",
            "network_node_labels",
            "network_edge_labels",
            "number_of_agents",
        ]:
            del requested_logs[name]

    if not isleconfig.save_network:
        for name in [
            "unweighted_network_data",
            "network_node_labels",
            "network_edge_labels",
            "number_of_agents",
        ]:
            del requested_logs[name]

    assert "number_riskmodels" in requested_logs

    """Configure log directory and ensure that the directory exists"""
    dir_prefix = "/data/"
    directory = os.getcwd() + dir_prefix
    try:  # Here it is checked whether the directory to collect the results exists or not. If not it is created.
        os.stat(directory)
    except FileNotFoundError:
        os.mkdir(directory)

    """Clear old dict saving files (*_history_logs.dat)"""
    for i in riskmodels:
        filename = os.getcwd() + dir_prefix + nums[str(i)] + "_history_logs.dat"
        if os.path.exists(filename):
            os.remove(filename)

    """Setup of the simulations"""
    # Here the setup for the simulation is done.
    # Since this script is used to carry out simulations in the cloud will usually have more than 1 replication.
    setup = SetupSim()
    [
        general_rc_event_schedule,
        general_rc_event_damage,
        np_seeds,
        random_seeds,
    ] = setup.obtain_ensemble(replications)
    # never save simulation state in ensemble runs (resuming is impossible anyway)
    save_iter = isleconfig.simulation_parameters["max_time"] + 2

    for i in riskmodels:
        # In this loop the parameters, schedules and random seeds for every run are prepared. Different risk models will
        # be run with the same schedule, damage size and random seed for a fair comparison.

        # Here the parameters used for the simulation are loaded. Clone is needed otherwise all the runs will be carried
        # out with the last number of the loop.
        simulation_parameters = copy.copy(parameters)
        # Since we want to obtain ensembles for different number of risk models, we vary the number of risks models.
        simulation_parameters["no_riskmodels"] = i
        # Here is assembled each job with the corresponding: simulation parameters, time events, damage events, seeds,
        # simulation state save interval (never, i.e. longer than max_time), and list of requested logs.
        job = [
            m(
                simulation_parameters,
                general_rc_event_schedule[x],
                general_rc_event_damage[x],
                np_seeds[x],
                random_seeds[x],
                save_iter,
                list(requested_logs.keys()),
            )
            for x in range(replications)
        ]
        jobs.append(job)  # All jobs are collected in the jobs list.

    """Here the jobs are submitted"""

    with Session(host=hostname, default_cb_to_stdout=True) as sess:

        for job in jobs:
            # If there are 4 risk models jobs will be a list with 4 elements.

            """Run simulation and obtain result"""
            result = sess.submit(job)

            """Find number of riskmodels from log"""
            delistified_result = [listify.delistify(list(res)) for res in result]
            nrm = delistified_result[0]["number_riskmodels"]

            """These are the files created to collect the results"""
            wfiles_dict = {}

            logfile_dict = {}

            for name in requested_logs.keys():
                if "rc_event" in name or "number_riskmodels" in name:
                    logfile_dict[name] = (
                        os.getcwd()
                        + dir_prefix
                        + "check_"
                        + str(nums[str(nrm)])
                        + requested_logs[name]
                    )
                elif "firms_cash" in name:
                    logfile_dict[name] = (
                        os.getcwd()
                        + dir_prefix
                        + "record_"
                        + str(nums[str(nrm)])
                        + requested_logs[name]
                    )
                else:
                    logfile_dict[name] = (
                        os.getcwd()
                        + dir_prefix
                        + str(nums[str(nrm)])
                        + requested_logs[name]
                    )

            # with ... as would be awkward here, so use try ... finally
            try:
                for name in logfile_dict:
                    wfiles_dict[name] = open(logfile_dict[name], "w")

                """Recreate logger object locally and save logs"""

                """Create local object"""
                log = logger.Logger()

                for i in range(len(job)):
                    """Populate logger object with logs obtained from remote simulation run"""
                    log.restore_logger_object(list(result[i]))

                    """Save logs as dict (to <num>_history_logs.dat)"""
                    log.save_log(True)
                    if isleconfig.save_network:
                        log.save_network_data(ensemble=True)

                    """Save logs as individual files"""
                    for name in logfile_dict:
                        wfiles_dict[name].write(str(delistified_result[i][name]) + "\n")

            finally:
                """Once the data is stored in disk the files are closed"""
                for name in logfile_dict:
                    if name in wfiles_dict:
                        wfiles_dict[name].close()
                        del wfiles_dict[name]


if __name__ == "__main__":
    host = None
    if len(sys.argv) > 1:
        host = sys.argv[1]  # The server is passed as an argument.
    rake(host)
