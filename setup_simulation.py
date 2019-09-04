"""Class to set up event schedules for reproducible simulation replications.
   Event schedule sets are written to files and include event schedules for every replication as dictionaries in a list.
   Every event schedule dictionary has:
      - event_times: list of list of int - iteration periods of risk events in each category
      - event_damages: list of list of float (0, 1) - damage as share of possible damage for each risk event
      - num_categories: int - number of risk categories
      - np_seed: int - numpy module random seed
      - random_seed: int - random module random seed
   A simulation given event schedule dictionary d should be set up like so:
        simulation.rc_event_schedule = d["event_times"]
        simulation.rc_event_damages = d["event_damages"]
        np.random.seed(d["np_seed"])
        random.random.seed(d["np_seed"])
    """

import math
from typing import MutableSequence, Tuple
import os
import pickle
import scipy.stats

import isleconfig
from distributiontruncated import TruncatedDistWrapper


class SetupSim:
    def __init__(self):

        self.simulation_parameters = isleconfig.simulation_parameters

        """parameters of the simulation setup"""
        self.max_time = self.simulation_parameters["max_time"]
        self.no_categories = self.simulation_parameters["no_categories"]

        """set distribution"""
        # It is assumed that the damages of the catastrophes are drawn from a truncated Pareto distribution.
        non_truncated = scipy.stats.pareto(b=2, loc=0, scale=0.25)
        self.damage_distribution = TruncatedDistWrapper(
            lower_bound=0.25, upper_bound=1.0, dist=non_truncated
        )
        self.cat_separation_distribution = scipy.stats.expon(
            0, self.simulation_parameters["event_time_mean_separation"]
        )  # It is assumed that the time between catastrophes is exponentially distributed.

        """"random seeds"""
        self.np_seed = []
        self.random_seed = []
        self.general_rc_event_schedule = []
        self.general_rc_event_damage = []
        self.filepath = "risk_event_schedules.islestore"
        self.overwrite = False
        self.replications = None

    def schedule(
        self, replications: int
    ) -> Tuple[
        MutableSequence[MutableSequence[int]], MutableSequence[MutableSequence[float]]
    ]:
        for i in range(replications):
            # In this list will be stored the lists of times when there will be catastrophes for every category of the
            # model during a single run. ([[times for C1],[times for C2],[times for C3],[times for C4]])
            rc_event_schedule = []
            # In this list will be stored the lists of catastrophe damages for every category of the model during a
            # single run. ([[damages for C1],[damages for C2],[damages for C3],[damages for C4]])
            rc_event_damage = []
            for j in range(self.no_categories):
                # In this list will be stored the times when there will be a catastrophe in a particular category.
                event_schedule = []
                # In this list will be stored the damages of a catastrophe related to a particular category.
                event_damage = []
                total = 0
                while total < self.max_time:
                    separation_time = self.cat_separation_distribution.rvs()
                    # Note: the ceil of an exponential distribution is just a geometric distribution
                    total += int(math.ceil(separation_time))
                    if total < self.max_time:
                        event_schedule.append(total)
                        event_damage.append(self.damage_distribution.rvs())
                rc_event_schedule.append(event_schedule)
                rc_event_damage.append(event_damage)

            self.general_rc_event_schedule.append(rc_event_schedule)
            self.general_rc_event_damage.append(rc_event_damage)

        return self.general_rc_event_schedule, self.general_rc_event_damage

    def seeds(self, replications: int):
        # This method sets (and returns) the seeds required for an ensemble of replications of the model.
        # The argument (replications) is the number of replications.
        """draw random variates for random seeds"""
        for i in range(replications):
            np_seed, random_seed = scipy.stats.randint.rvs(0, 2 ** 32 - 1, size=2)
            self.np_seed.append(np_seed)
            self.random_seed.append(random_seed)

        return self.np_seed, self.random_seed

    def store(self):
        # This method stores in a file the the schedules and random seeds required for an ensemble of replications of
        # the model. The number of replications is calculated from the length of the exisiting values.
        # With the information stored it is possible to replicate the entire behavior of the ensemble at a later time.
        event_schedules = []
        if not (
            len(self.np_seed)
            == len(self.random_seed)
            == len(self.general_rc_event_damage)
            == len(self.general_rc_event_schedule)
        ):
            raise ValueError("Required data not all the same lenght, can't store")
        replications = len(self.np_seed)

        for i in range(replications):
            """pack to dict"""
            d = {}
            d["np_seed"] = self.np_seed[i]
            d["random_seed"] = self.random_seed[i]
            d["event_times"] = self.general_rc_event_schedule[i]
            d["event_damages"] = self.general_rc_event_damage[i]
            d["num_categories"] = self.simulation_parameters["no_categories"]
            event_schedules.append(d)

        """ ensure that logging directory exists"""
        if not os.path.isdir("data"):
            if os.path.exists("data"):
                raise Exception(
                    "./data exists as regular file. "
                    "This filename is required for the logging and event schedule directory"
                )
            os.makedirs("data")

        # If we are avoiding overwriting, check if the file to write to exist
        if not self.overwrite:
            if os.path.exists("data/" + self.filepath):
                raise ValueError(
                    f"File {'./data/' + self.filepath} already exists and we are not overwriting."
                )

        """Save the initial values"""
        with open("./data/" + self.filepath, "wb") as wfile:
            pickle.dump(event_schedules, wfile, protocol=pickle.HIGHEST_PROTOCOL)

    def recall(self):
        if not (
            self.np_seed
            == self.random_seed
            == self.general_rc_event_schedule
            == self.general_rc_event_damage
            == []
        ):
            raise ValueError("Some of the data to be recalled already exists")
        with open("./data/" + self.filepath, "rb") as rfile:
            event_schedules = pickle.load(rfile)
        self.replications = len(event_schedules)
        for initial_values in event_schedules:
            self.np_seed.append(initial_values["np_seed"])
            self.random_seed.append(initial_values["random_seed"])
            self.general_rc_event_schedule.append(initial_values["event_times"])
            self.general_rc_event_damage.append(initial_values["event_damages"])
            self.simulation_parameters["no_categories"] = initial_values[
                "num_categories"
            ]

    def obtain_ensemble(
        self, replications: int, filepath: str = None, overwrite: bool = False
    ) -> Tuple:
        # This method returns all the information (schedules and seeds) required to run an ensemble of simulations of
        # the model. Since it also stores the information in a file it will be possible to replicate the ensemble at a
        # later time. The argument (replications) is the number of replications.
        # This method will be called either form ensemble.py or start.py
        if filepath is not None:
            self.filepath = self.to_filename(filepath)
        self.overwrite = overwrite
        if not isleconfig.replicating:
            # Not replicating another run, so we are writing to the file given
            self.replications = replications
            if filepath is None and not self.overwrite:
                print(
                    "No explicit path given, automatically overwriting default path for initial state"
                )
                self.overwrite = True
            self.schedule(replications)
            self.seeds(replications)

            self.store()
        else:
            # Replicating anothe run, so we are reading from the file given
            if filepath is not None:
                self.recall()
                if replications != self.replications:
                    raise ValueError(
                        f"Found {self.replications} replications in given file, expected {replications}."
                    )
            else:
                # Could read from default file, seems like a bad idea though.
                raise ValueError(
                    "Simulation is set to replicate but no replicid has been given"
                )

        return (
            self.general_rc_event_schedule,
            self.general_rc_event_damage,
            self.np_seed,
            self.random_seed,
        )

    @staticmethod
    def to_filename(filepath: str) -> str:
        if len(filepath) >= 10 and filepath[-10:] == ".islestore":
            return filepath
        else:
            return filepath + ".islestore"
