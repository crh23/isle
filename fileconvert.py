import sys
import os

import hickle
import numpy as np

import logger

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


def convert(prefix: str):
    dir_prefix = "/data/"
    if not prefix.endswith("_full_logs.hdf"):
        file = "data/" + prefix + "_full_logs.hdf"
    else:
        file = "data/" + prefix
    results_dict = hickle.load(file)

    found_logs = list(results_dict.keys())
    logfile_dict = {}
    for name in found_logs:
        if "rc_event" in name or "number_riskmodels" in name:
            logfile_dict[name] = (
                os.getcwd() + dir_prefix + "check_" + prefix + requested_logs[name]
            )
        elif "firms_cash" in name:
            logfile_dict[name] = (
                os.getcwd() + dir_prefix + "record_" + prefix + requested_logs[name]
            )
        else:
            logfile_dict[name] = (
                os.getcwd() + dir_prefix + "data_" + prefix + requested_logs[name]
            )
    for name in logfile_dict:
        with open(logfile_dict[name], "w") as rfile:
            this_data = results_dict[name]
            if isinstance(this_data, np.ndarray):
                if this_data.ndim == 0:
                    this_data = [this_data]
                for replication_data in this_data:
                    rfile.write(repr(replication_data.tolist()) + "\n")
            else:
                if not isinstance(this_data, list):
                    raise ValueError("Data is neither a list nor an array")
                for replication_data in this_data:
                    rfile.write(repr(replication_data) + "\n")


if __name__ == "__main__":
    # filename = None
    # if len(sys.argv) > 1:
    #     # The server is passed as an argument.
    #     filename = sys.argv[1]
    # else:
    #     raise ValueError("No filename or prefix given")
    filename = "ensemble1"
    convert(filename)
