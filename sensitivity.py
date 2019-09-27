"""
A modification of ensemble.py to do sensitivity analysis using SALib
"""
import sys
import os
from typing import Dict
import importlib

import numpy as np
import isleconfig
import start
import setup_simulation
import calibration_statistic


def rake(hostname=None, summary: callable = None, use_sandman: bool = False):
    """
    Uses the sandman2 api to run multiple replications of multiple configurations of the simulation.
    If hostname=None, runs locally. Otherwise, make sure environment variable SANDMAN_KEY_ID and SANDMAN_KEY_SECRET
    are set.
    Args:
        hostname: The remote server to run the job on
        summary: The summary statistic (function) to apply to the results
        use_sandman: if True, uses sandman, otherwise uses multiprocessing (faster if running very many simulations
                        locally)
    """

    # TODO: RM
    np.seterr(all="raise")

    if importlib.util.find_spec("hickle") is None:
        raise ModuleNotFoundError("hickle not found but required for saving logs")

    if hostname is None:
        print("Running ensemble locally")
    else:
        if use_sandman:
            print(f"Running ensemble on {hostname}")
        else:
            raise ValueError("use_sandman is False, but hostname is given")
    """Configure the parameter sets to run"""
    default_parameters: Dict = isleconfig.simulation_parameters
    parameter_list = None

    ###################################################################################################################
    # This section should be freely modified to determine the experiment
    # parameters should be a list of (hashable) lables for the settings, which parameter_list should be a list of.

    import SALib.util
    import SALib.sample.morris

    problem = SALib.util.read_param_file("isle_all_parameters.txt")
    param_values = SALib.sample.morris.sample(problem, N=problem["num_vars"] * 3)
    parameters = [tuple(row) for row in param_values]
    parameter_list = [
        {
            **default_parameters.copy(),
            "max_time": 2000,
            **{problem["names"][i]: row[i] for i in range(len(row))},
        }
        for row in param_values
    ]
    if parameter_list[1] == parameter_list[0]:
        raise RuntimeError("Parameter list appears to be homogenous!")

    ###################################################################################################################

    max_time = parameter_list[0]["max_time"]

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
        "insurance_contracts": "_insurance_contracts.dat",
        "reinsurance_contracts": "_reinsurance_contracts.dat",
        "unweighted_network_data": "_unweighted_network_data.dat",
        "network_node_labels": "_network_node_labels.dat",
        "network_edge_labels": "_network_edge_labels.dat",
        "number_of_agents": "_number_of_agents",
    }

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
    print("Setting up simulation")
    [
        general_rc_event_schedule,
        general_rc_event_damage,
        np_seeds,
        random_seeds,
    ] = setup.obtain_ensemble(len(parameter_list))

    n = len(parameter_list)
    m_params = list(
        zip(
            parameter_list,
            general_rc_event_schedule,
            general_rc_event_damage,
            np_seeds,
            random_seeds,
            [0] * n,
            [0] * n,
            [None] * n,
            [False] * n,
            [summary] * n,
        )
    )

    if use_sandman:
        import sandman2.api as sm

        print("Constructing sandman operation")
        m = sm.operation(start.main, include_modules=True)
        print("Assembling jobs")

        # Here is assembled each job with the corresponding: simulation parameters, time events, damage events, seeds,
        # simulation state save interval (never), and list of requested logs.

        # This is actually quite slow for large sets of jobs. Can't use mp.Pool due to unpickleability
        # Could use pathos or similar if we actually end up caring
        job = list(map(m, m_params))
        # # Split up into chunks so sandman server doesn't blow up
        # max_size = 71
        # job_lists = []
        # while len(job) > 0:
        #     job_lists.append(job[: min(max_size, len(job))])
        #     job = job[min(max_size, len(job)) :]
        """Here the jobs are submitted"""
        print("Jobs created, submitting")
        with sm.Session(host=hostname, default_cb_to_stdout=True) as sess:
            print("Starting job")
            # result = []
            # for job in job_lists:
            #     result += sess.submit(job)

            # Submit async so we can reattach with sess.get if something goes wrong locally
            task = sess.submit_async(job)

            task.wait()
            result = task.results
    else:
        # result = []
        # m_params.reverse()
        # for i, param_set in enumerate(m_params):
        #     result.append(start.main(param_set))
        import multiprocessing as mp

        print("Running multiprocessing pool")
        # set maxtasksperchild, otherwise it seems that garbage collection(?) misbehaves and we get huge memory usage
        with mp.Pool(maxtasksperchild=1) as pool:
            # Since the jobs are so big, chunksize=1 is best
            result = pool.map(start.main, m_params, chunksize=1)

    print("Job done, saving")
    result_dict = {t: r for t, r in zip(parameters, result)}
    start.save_summary([result_dict])


def analyse(data: dict):
    keylist = list(data.keys())
    # SALib expects a matrix, so we give it a matrix
    x = np.array(keylist)
    outputs = []
    for key_name in keylist:
        result_dict = data[key_name]
        result = [
            result_dict[name]
            for name in calibration_statistic.statistics
            if name in result_dict
        ]
        outputs.append(result)
        found_statistics = [
            name for name in calibration_statistic.statistics if name in result_dict
        ]
    y_full = np.array(outputs)

    import SALib.util
    import SALib.analyze.morris

    problem = SALib.util.read_param_file("isle_all_parameters.txt")
    outputs = {}
    for i, stat in enumerate(found_statistics):
        print(stat + ":")
        outputs[stat] = SALib.analyze.morris.analyze(
            problem, x, y_full[:, i], print_to_console=True
        )
    return outputs


if __name__ == "__main__":
    import hickle

    data = hickle.load("data/summary_statistics.hdf")
    result = analyse(data[0])
    hickle.dump(result, "data/sensitivity_analysis_results.hdf")
    # host = None
    # remote = False
    # if len(sys.argv) > 1:
    #     # The server is passed as an argument.
    #     host = sys.argv[1]
    #     remote = True
    # rake(
    #     host, summary="calibration_statistic.calculate_single", use_sandman=remote
    # )
    # # jobs = {"ensemble1" : "23a3f4e1",
    # #         "ensemble2" : "485f7221"}
    # # restore_jobs(jobs, host)
