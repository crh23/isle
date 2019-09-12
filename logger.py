"""Logging class. Handles records of a single simulation run. Can save and reload. """

import numpy as np
import listify
import os
import time

LOG_DEFAULT = (
    "total_cash total_excess_capital total_profitslosses total_contracts "
    "total_operational total_reincash total_reinexcess_capital total_reinprofitslosses "
    "total_reincontracts total_reinoperational total_catbondsoperational market_premium "
    "market_reinpremium cumulative_bankruptcies cumulative_market_exits cumulative_unrecovered_claims "
    "cumulative_claims insurance_firms_cash reinsurance_firms_cash market_diffvar "
    "rc_event_schedule_initial rc_event_damage_initial number_riskmodels insurance_contracts reinsurance_contracts "
    "unweighted_network_data network_node_labels network_edge_labels number_of_agents "
    "cumulative_bought_firms cumulative_nonregulation_firms insurance_cumulative_dividends "
    "reinsurance_cumulative_dividends"
).split(" ")

firm_level_logs = [
    "insurance_firms_cash",
    "reinsurance_firms_cash",
    "insurance_contracts",
    "reinsurance_contracts",
    "insurance_claims",
    "reinsurance_claims",
    "insurance_pls",
    "reinsurance_pls",
    "insurance_cumulative_premiums",
    "reinsurance_cumulative_premiums",
    "insurance_ratios",
    "reinsurance_ratios",
]


class Logger:
    def __init__(
        self,
        no_riskmodels=None,
        rc_event_schedule_initial=None,
        rc_event_damage_initial=None,
    ):
        """Constructor. Prepares history_logs atribute as dict for the logs. Records initial event schedule of
           simulation run.
            Arguments
                no_categories: Type int. number of peril regions.
                rc_event_schedule_initial: list of lists of int. Times of risk events by category
                rc_event_damage_initial: list of arrays (or lists) of float. Damage by peril for each category
                                         as share of total possible damage (maximum insured or excess).
            Returns class instance."""

        """Record number of riskmodels"""
        self.number_riskmodels = no_riskmodels

        """Record initial event schedule"""
        self.rc_event_schedule_initial = rc_event_schedule_initial
        self.rc_event_damage_initial = rc_event_damage_initial

        """Prepare history log dict"""
        self.history_logs = {}
        self.history_logs_to_save = []

        """Variables pertaining to insurance sector"""
        # TODO: should we not have `cumulative_bankruptcies` and
        #  `cumulative_market_exits` for both insurance firms and reinsurance firms?
        # `cumulative_claims`: Here are stored the total cumulative claims received
        # by the whole insurance sector until a certain time.
        insurance_sector = (
            "total_cash total_excess_capital total_profitslosses "
            "total_contracts total_operational cumulative_bankruptcies "
            "cumulative_market_exits cumulative_claims cumulative_unrecovered_claims "
            "cumulative_bought_firms cumulative_nonregulation_firms"
        ).split(" ")
        for _v in insurance_sector:
            self.history_logs[_v] = []

        for name in firm_level_logs:
            self.history_logs[name] = []

        """Variables pertaining to reinsurance sector"""
        self.history_logs["total_reincash"] = []
        self.history_logs["total_reinexcess_capital"] = []
        self.history_logs["total_reinprofitslosses"] = []
        self.history_logs["total_reincontracts"] = []
        self.history_logs["total_reinoperational"] = []

        """Variables pertaining to cat bonds"""
        self.history_logs["total_catbondsoperational"] = []

        """Variables pertaining to premiums"""
        self.history_logs["market_premium"] = []
        self.history_logs["market_reinpremium"] = []
        self.history_logs["market_diffvar"] = []

        "Network Data Logs to be stored in separate file"
        self.network_data = {}
        self.network_data["unweighted_network_data"] = []
        self.network_data["network_node_labels"] = []
        self.network_data["network_edge_labels"] = []
        self.network_data["number_of_agents"] = []

        self.history_logs["unweighted_network_data"] = []
        self.history_logs["network_node_labels"] = []
        self.history_logs["network_edge_labels"] = []
        self.history_logs["number_of_agents"] = []
        self.history_logs["insurance_cumulative_dividends"] = []
        self.history_logs["reinsurance_cumulative_dividends"] = []

    def record_data(self, data_dict):
        """Method to record data for one period
            Arguments
                data_dict: Type dict. Data with the same keys as are used in self.history_log().
            Returns None."""
        for key in data_dict.keys():
            if key in firm_level_logs:
                # These are stored per-firm
                if not len(data_dict[key]) == len(self.history_logs[key]):
                    raise RuntimeError(
                        f"Log {key} passed to logger has different number of firms to those already in"
                        f" log - {len(data_dict[key])} passed, {len(self.history_logs[key])} expected"
                    )
                for i in range(len(data_dict[key])):
                    self.history_logs[key][i].append(data_dict[key][i])
            else:
                self.history_logs[key].append(data_dict[key])

    def obtain_log(self, requested_logs=None) -> dict:
        if requested_logs is None:
            requested_logs = self.history_logs.keys()
        """Method to transfer entire log (self.history_log as well as risk event schedule). This is
           used to transfer the log to master core from work cores in ensemble runs in the cloud.
            No arguments.
            Returns list (listified dict)."""

        """Include environment variables (number of risk models and risk event schedule)"""
        self.history_logs["number_riskmodels"] = self.number_riskmodels
        self.history_logs["rc_event_damage_initial"] = self.rc_event_damage_initial
        self.history_logs["rc_event_schedule_initial"] = self.rc_event_schedule_initial

        """Parse logs to be returned"""
        log = {name: self.history_logs[name] for name in requested_logs}

        """Convert to list and return"""
        # return listify.listify(log)
        return log

    def restore_logger_object(self, log):
        """Method to restore logger object. A log can be restored later. It can also be restored
           on a different machine. This is useful in the case of ensemble runs to move the log to
           the master node from the computation nodes.
            Arguments:
                log - listified dict - The log. This must be a list of dict values plus the dict
                                        keys in the last element. It should have been created by
                                        listify.listify()
            Returns None."""

        """Restore dict"""
        if not isinstance(log, dict):
            log = listify.delistify(log)
        try:
            self.network_data["unweighted_network_data"] = log[
                "unweighted_network_data"
            ]
            self.network_data["network_node_labels"] = log["network_node_labels"]
            self.network_data["network_edge_labels"] = log["network_edge_labels"]
            self.network_data["number_of_agents"] = log["number_of_agents"]
            del (
                log["number_of_agents"],
                log["network_edge_labels"],
                log["network_node_labels"],
                log["unweighted_network_data"],
            )
        except KeyError:
            pass

        """Extract environment variables (number of risk models and risk event schedule)"""
        self.rc_event_schedule_initial = log["rc_event_schedule_initial"]
        self.rc_event_damage_initial = log["rc_event_damage_initial"]
        self.number_riskmodels = log["number_riskmodels"]

        """Restore history log"""
        self.history_logs_to_save.append(log)
        self.history_logs = log

    def save_log(self, ensemble_run: bool, prefix: str = "") -> None:
        """Method to save log to disk of local machine. Distinguishes single and ensemble runs.
           Is called at the end of the replication (if at all).
            Arguments:
                ensemble_run: Type bool. Is this an ensemble run (true) or not (false).
                prefix: Type str. The prefix to prepend to the filename
            Returns None."""
        # TODO: remove if not required
        """Prepare writing tasks"""
        if ensemble_run:
            to_log = self.replication_log_prepare(prefix)
        else:
            to_log = self.single_log_prepare()

        """Write to disk"""
        for filename, data, operation_character in to_log:
            with open(filename, operation_character) as wfile:
                wfile.write(str(data) + "\n")

    def replication_log_prepare(self, prefix, position: int = None):
        """Method to prepare writing tasks for ensemble run saving.
            No arguments
            Returns list of tuples with three elements each.
                    Element 1: filename
                    Element 2: data structure to save
                    Element 3: operation parameter (w-write or a-append)."""
        if position is not None:
            data = (position, self.history_logs)
        else:
            data = self.history_logs
        to_log = [("data/" + "full_" + prefix + "_history_logs.dat", data, "a")]
        return to_log

    def single_log_prepare(self, prefix="single"):
        """Method to prepare writing tasks for single run saving.
            No arguments
            Returns list of tuples with three elements each.
                    Element 1: filename
                    Element 2: data structure to save
                    Element 3: operation parameter (w-write or a-append)."""
        to_log = []
        filename = "data/" + prefix + "_history_logs.dat"
        backupfilename = (
            "data/"
            + prefix
            + "_history_logs_old_"
            + time.strftime("%Y_%b_%d_%H_%M")
            + ".dat"
        )
        if os.path.exists(filename):
            os.rename(filename, backupfilename)
        for history_log in self.history_logs_to_save:
            to_log.append((filename, history_log, "a"))
        return to_log

    def save_network_data(self, ensemble, prefix=""):
        """Method to save network data to its own file.
            Accepts:
                ensemble: Type Boolean. Saves to files based on number risk models.
            No return values."""
        import pickle

        if ensemble:
            network_logs = [
                ("data/" + prefix + "_network_data.pkl", self.network_data, "r+b")
            ]
            for filename, data, operation_character in network_logs:
                with open(filename, operation_character) as file:
                    list_ = pickle.load(file)
                    list_.append(data)
                    file.truncate(0)
                    pickle.dump(list_, file)
                    # wfile.write(str(data) + "\n")
        else:
            with open("data/network_data.pkl", "wb") as wfile:
                pickle.dump((self.network_data, self.rc_event_schedule_initial), wfile)
                # wfile.write(str(self.network_data) + "\n")
                # wfile.write(str(self.rc_event_schedule_initial) + "\n")

    def add_firm(self, firm_type: str):
        """Notifies the logger of a new firm, so blank data can be added to firm-level logs"""
        keys = [key for key in firm_level_logs if key.startswith(firm_type)]
        for key in keys:
            if len(self.history_logs[key]) > 0:
                zeroes_to_append = list(
                    np.zeros(len(self.history_logs[key][0]), dtype=float)
                )
            else:
                zeroes_to_append = []
            self.history_logs[key].append(zeroes_to_append)
