from distributiontruncated import TruncatedDistWrapper
import visualization_network
import numpy as np
import scipy.stats
import math
import sys, pdb
import isleconfig
import random
import copy
import logger
import warnings
import utils


class InsuranceSimulation:
    """ Simulation object that is responsible for handling all aspects of the world.

    Tracks all agents (firms, catbonds) as well as acting as the insurance market. Iterates other objects,
    distributes risks, pays premiums, recieves claims, tracks and inflicts perils, etc. Also contains functionality
    to log the state of the simulation.

    Each insurer is given a set of inaccuracy values, on for each category. This is a factor that is inserted when the
    insurer calculates the expected damage from a catastophe. In the current configuration, this uses the
    riskmodel_inaccuracy_parameter in the configuration - a randomly chosen category has its inaccuracy set to the
    inverse of that parameter, and the others are set to that parameter."""

    def __init__(
        self,
        override_no_riskmodels,
        replic_ID,
        simulation_parameters,
        rc_event_schedule,
        rc_event_damage,
        damage_distribution=TruncatedDistWrapper(
            lower_bound=0.25,
            upper_bound=1.0,
            dist=scipy.stats.pareto(b=2, loc=0, scale=0.25),
        ),
    ):
        """Initialises the simulation (Called from start.py)
                    Accepts:
                        override_no_riskmodels: Boolean determining if number of risk models should be overwritten
                        replic_ID: Integer, used if want to replicate data over multiple runs
                        simulation parameters: DataDict from isleconfig
                        rc_event_schedule: List of when event will occur, allows for replication
                        re_event_damage: List of severity of each event, allows for replication"""

        "Override one-riskmodel case (this is to ensure all other parameters are truly identical for comparison runs)"
        if override_no_riskmodels:
            simulation_parameters["no_riskmodels"] = override_no_riskmodels
        # QUERY: why do we keep duplicates of so many simulation parameters (and then not use many of them)?
        self.number_riskmodels = simulation_parameters["no_riskmodels"]

        # save parameters
        if (replic_ID is None) or isleconfig.force_foreground:
            self.background_run = False
        else:
            self.background_run = True
        self.replic_ID = replic_ID
        self.simulation_parameters = simulation_parameters

        "Unpacks parameters and sets distributions"
        self.damage_distribution = damage_distribution

        self.catbonds_off = simulation_parameters["catbonds_off"]
        self.reinsurance_off = simulation_parameters["reinsurance_off"]
        # TODO: research whether this is accurate, is it different for different types of catastrophy?
        self.cat_separation_distribution = scipy.stats.expon(
            0, simulation_parameters["event_time_mean_separation"]
        )
        self.risk_factor_lower_bound = simulation_parameters["risk_factor_lower_bound"]
        self.risk_factor_spread = (
            simulation_parameters["risk_factor_upper_bound"]
            - self.risk_factor_lower_bound
        )
        if simulation_parameters["risk_factors_present"]:
            self.risk_factor_distribution = scipy.stats.uniform(
                loc=self.risk_factor_lower_bound, scale=self.risk_factor_spread
            )
        else:
            self.risk_factor_distribution = utils.constant(loc=1.0)
        # self.risk_value_distribution = scipy.stats.uniform(loc=100, scale=9900)
        self.risk_value_distribution = utils.constant(loc=1000)

        risk_factor_mean = self.risk_factor_distribution.mean()

        "set initial market price (normalized, i.e. must be multiplied by value or excess-deductible)"
        if self.simulation_parameters["expire_immediately"]:
            assert self.cat_separation_distribution.dist.name == "expon"
            expected_damage_frequency = 1 - scipy.stats.poisson(
                self.simulation_parameters["mean_contract_runtime"]
                / self.simulation_parameters["event_time_mean_separation"]
            ).pmf(0)
        else:
            expected_damage_frequency = (
                self.simulation_parameters["mean_contract_runtime"]
                / self.cat_separation_distribution.mean()
            )
        self.norm_premium = (
            expected_damage_frequency
            * self.damage_distribution.mean()
            * risk_factor_mean
            * (1 + self.simulation_parameters["norm_profit_markup"])
        )

        self.market_premium = self.norm_premium
        self.reinsurance_market_premium = self.market_premium
        # TODO: is this problematic as initial value? (later it is recomputed in every iteration)

        self.total_no_risks = simulation_parameters["no_risks"]

        "Set up monetary system (should instead be with the customers, if customers are modeled explicitly)"
        self.money_supply = self.simulation_parameters["money_supply"]
        self.obligations = []
        # QUERY Why is this a property of the simulation rather than of the obligated parties?

        "set up risk categories"
        # QUERY What do risk categories represent? Different types of catastrophes?
        self.riskcategories = list(range(self.simulation_parameters["no_categories"]))
        self.rc_event_schedule = []
        self.rc_event_damage = []
        self.rc_event_schedule_initial = []
        # For debugging (cloud debugging) purposes is good to store the initial schedule of catastrophes
        # and damages that will be use in a single run of the model.
        self.rc_event_damage_initial = []
        if (
            rc_event_schedule is not None and rc_event_damage is not None
        ):  # If we have schedules pass as arguments we used them.
            self.rc_event_schedule = copy.copy(rc_event_schedule)
            self.rc_event_schedule_initial = copy.copy(rc_event_schedule)

            self.rc_event_damage = copy.copy(rc_event_damage)
            self.rc_event_damage_initial = copy.copy(rc_event_damage)
        else:  # Otherwise the schedules and damages are generated.
            raise Exception("No event schedules and damages supplied")

        "Set up risks"
        risk_value_mean = self.risk_value_distribution.mean()

        # QUERY: What are risk factors? Are "risk_factor" values other than one meaningful at present?
        rrisk_factors = self.risk_factor_distribution.rvs(
            size=self.simulation_parameters["no_risks"]
        )
        rvalues = self.risk_value_distribution.rvs(
            size=self.simulation_parameters["no_risks"]
        )
        rcategories = np.random.randint(
            0,
            self.simulation_parameters["no_categories"],
            size=self.simulation_parameters["no_risks"],
        )
        self.risks = [
            {
                "risk_factor": rrisk_factors[i],
                "value": rvalues[i],
                "category": rcategories[i],
                "owner": self,
            }
            for i in range(self.simulation_parameters["no_risks"])
        ]

        self.risks_counter = [0, 0, 0, 0]

        for item in self.risks:
            self.risks_counter[item["category"]] = (
                self.risks_counter[item["category"]] + 1
            )

        self.inaccuracy = self._get_all_riskmodel_combinations(
            self.simulation_parameters["no_categories"],
            self.simulation_parameters["riskmodel_inaccuracy_parameter"],
        )

        self.inaccuracy = random.sample(
            self.inaccuracy, self.simulation_parameters["no_riskmodels"]
        )

        risk_model_configurations = [
            {
                "damage_distribution": self.damage_distribution,
                "expire_immediately": self.simulation_parameters["expire_immediately"],
                "cat_separation_distribution": self.cat_separation_distribution,
                "norm_premium": self.norm_premium,
                "no_categories": self.simulation_parameters["no_categories"],
                "risk_value_mean": risk_value_mean,
                "risk_factor_mean": risk_factor_mean,
                "norm_profit_markup": self.simulation_parameters["norm_profit_markup"],
                "margin_of_safety": self.simulation_parameters[
                    "riskmodel_margin_of_safety"
                ],
                "var_tail_prob": self.simulation_parameters[
                    "value_at_risk_tail_probability"
                ],
                "inaccuracy_by_categ": self.inaccuracy[i],
            }
            for i in range(self.simulation_parameters["no_riskmodels"])
        ]

        "Setting up agents (to be done from start.py)"
        # QUERY: What is agent_parameters["insurancefirm"] meant to be? Is it a list of the parameters for the existing
        #  firms (why can't we just get that from the instances of InsuranceFirm) or a list of the *possible* parameter
        #  values for insurance firms (in which case why does it have the length it does)?
        self.agent_parameters = {"insurancefirm": [], "reinsurancefirm": []}
        self.initialize_agent_parameters(
            "insurancefirm", simulation_parameters, risk_model_configurations
        )
        self.initialize_agent_parameters(
            "reinsurancefirm", simulation_parameters, risk_model_configurations
        )

        "Agent lists"
        self.reinsurancefirms = []
        self.insurancefirms = []
        self.catbonds = []

        "Lists of agent weights"
        self.insurers_weights = {}
        self.reinsurers_weights = {}

        "List of reinsurance risks offered for underwriting"
        self.reinrisks = []
        self.not_accepted_reinrisks = []

        "Cumulative variables for history and logging"
        self.cumulative_bankruptcies = 0
        self.cumulative_market_exits = 0
        self.cumulative_unrecovered_claims = 0.0
        self.cumulative_claims = 0.0

        "Lists for logging history"
        self.logger = logger.Logger(
            no_riskmodels=simulation_parameters["no_riskmodels"],
            rc_event_schedule_initial=self.rc_event_schedule_initial,
            rc_event_damage_initial=self.rc_event_damage_initial,
        )

        self.insurance_models_counter = np.zeros(
            self.simulation_parameters["no_categories"]
        )
        self.reinsurance_models_counter = np.zeros(
            self.simulation_parameters["no_categories"]
        )

    def initialize_agent_parameters(
        self, firmtype, simulation_parameters, risk_model_configurations
    ):
        """General function for initialising the agent parameters
            Takes the firm type as argument, also needing sim params and risk configs
              Creates the agent parameters of both firm types for the initial number specified in isleconfig.py
                Returns None"""
        if firmtype == "insurancefirm":
            self.insurer_id_counter = 0
            no_firms = simulation_parameters["no_insurancefirms"]
            initial_cash = "initial_agent_cash"
            reinsurance_level_lowerbound = simulation_parameters[
                "insurance_reinsurance_levels_lower_bound"
            ]
            reinsurance_level_upperbound = simulation_parameters[
                "insurance_reinsurance_levels_upper_bound"
            ]

        elif firmtype == "reinsurancefirm":
            self.reinsurer_id_counter = 0
            no_firms = simulation_parameters["no_reinsurancefirms"]
            initial_cash = "initial_reinagent_cash"
            reinsurance_level_lowerbound = simulation_parameters[
                "reinsurance_reinsurance_levels_lower_bound"
            ]
            reinsurance_level_upperbound = simulation_parameters[
                "reinsurance_reinsurance_levels_upper_bound"
            ]
        else:
            raise ValueError(f"Firm type {firmtype} not recognised")

        for i in range(no_firms):
            if simulation_parameters["static_non-proportional_reinsurance_levels"]:
                reinsurance_level = simulation_parameters[
                    "default_non-proportional_reinsurance_deductible"
                ]
            else:
                reinsurance_level = np.random.uniform(
                    reinsurance_level_lowerbound, reinsurance_level_upperbound
                )

            riskmodel_config = risk_model_configurations[
                i % len(risk_model_configurations)
            ]
            self.agent_parameters[firmtype].append(
                {
                    "id": self.get_unique_insurer_id(),
                    "initial_cash": simulation_parameters[initial_cash],
                    "riskmodel_config": riskmodel_config,
                    "norm_premium": self.norm_premium,
                    "profit_target": simulation_parameters["norm_profit_markup"],
                    "initial_acceptance_threshold": simulation_parameters[
                        "initial_acceptance_threshold"
                    ],
                    "acceptance_threshold_friction": simulation_parameters[
                        "acceptance_threshold_friction"
                    ],
                    "reinsurance_limit": simulation_parameters["reinsurance_limit"],
                    "non-proportional_reinsurance_level": reinsurance_level,
                    "capacity_target_decrement_threshold": simulation_parameters[
                        "capacity_target_decrement_threshold"
                    ],
                    "capacity_target_increment_threshold": simulation_parameters[
                        "capacity_target_increment_threshold"
                    ],
                    "capacity_target_decrement_factor": simulation_parameters[
                        "capacity_target_decrement_factor"
                    ],
                    "capacity_target_increment_factor": simulation_parameters[
                        "capacity_target_increment_factor"
                    ],
                    "interest_rate": simulation_parameters["interest_rate"],
                }
            )

    def add_agents(self, agent_class, agent_class_string):
        # TODO: implement this to merge build_agents and accept_agents
        pass

    def build_agents(
        self, agent_class, agent_class_string, parameters, agent_parameters
    ):
        """Method for building new agents, only used for re/insurance firms. Loops through the agent parameters for each
            initialised agent to create an instance of them using re/insurancefirm.
            Accepts:
                Agent_class: class of agent, either InsuranceFirm or ReinsuranceFirm.
                agent_class_string: String Type containing string of agent class. Not used.
                parameters: DataDict, contains config parameters.
                agent_parameters: DataDict of agent parameters.
            Returns:
                agents: List Type, list of agent class instances created by loop"""
        assert parameters == self.simulation_parameters
        agents = []
        for ap in agent_parameters:
            agents.append(agent_class(parameters, ap))
        return agents

    def accept_agents(self, agent_class_string, agents, time=0):
        """Method to 'accept' agents in that it adds agent to relevant list of agents kept by simulation
            instance, also adds agent to logger. Also takes created agents initial cash out of economy.
            Accepts:
                agent_class_string: String Type.
                agents: List type of agent class instances.
                agent_group: List type of agent class instances.
                time: Integer type, not used
            Returns:
                None"""
        if agent_class_string == "insurancefirm":
            try:
                self.insurancefirms += agents
                self.insurancefirms_group = agents
            except:  # QUERY: Why?
                print(sys.exc_info())
                pdb.set_trace()
            # fix self.history_logs['individual_contracts'] list
            for agent in agents:
                self.logger.add_insurance_agent()
            # remove new agent cash from simulation cash to ensure stock flow consistency
            total_new_agent_cash = sum([agent.cash for agent in agents])
            self._reduce_money_supply(total_new_agent_cash)
        elif agent_class_string == "reinsurancefirm":
            try:
                self.reinsurancefirms += agents
                self.reinsurancefirms_group = agents
            except:
                print(sys.exc_info())
                pdb.set_trace()
            # remove new agent cash from simulation cash to ensure stock flow consistency
            total_new_agent_cash = sum([agent.cash for agent in agents])
            self._reduce_money_supply(total_new_agent_cash)
        elif agent_class_string == "catbond":
            try:
                self.catbonds += agents
            except:
                print(sys.exc_info())
                pdb.set_trace()
        else:
            raise ValueError(f"Error: Unexpected agent class used {agent_class_string}")

    def delete_agents(self, agent_class_string, agents):
        """Method for deleting catbonds as it is only agent that is allowed to be removed
            alters lists of catbonds
              Returns none"""
        if agent_class_string == "catbond":
            for agent in agents:
                self.catbonds.remove(agent)
        else:
            raise ValueError(
                f"Trying to remove unremovable agent, type: {agent_class_string}"
            )

    def iterate(self, t):
        """Function that is called from start.py for each iteration that settles obligations, capital then reselects
            risks for the insurance and reinsurance companies to evaluate. Firms are then iterated through to accept
              new risks, pay obligations, increase capacity etc.
           Accepts:
                t: Integer, current time step
           Returns None"""
        if isleconfig.verbose:
            print()
            print(t, ": ", len(self.risks))
        if isleconfig.showprogress:
            print(f"\rTime: {t}", end="")

        self.reset_pls()

        # adjust market premiums
        sum_capital = sum([agent.get_cash() for agent in self.insurancefirms])
        self._adjust_market_premium(capital=sum_capital)
        sum_capital = sum([agent.get_cash() for agent in self.reinsurancefirms])
        self._adjust_reinsurance_market_premium(capital=sum_capital)

        # Pay obligations
        self._effect_payments(t)

        # identify perils and effect claims
        for categ_id in range(len(self.rc_event_schedule)):
            if (
                self.rc_event_schedule[categ_id]
                and self.rc_event_schedule[categ_id][0] < t
            ):
                warnings.warn(
                    "Something wrong; past events not deleted", RuntimeWarning
                )
            if (
                len(self.rc_event_schedule[categ_id]) > 0
                and self.rc_event_schedule[categ_id][0] == t
            ):
                self.rc_event_schedule[categ_id] = self.rc_event_schedule[categ_id][1:]
                damage_extent = copy.copy(
                    self.rc_event_damage[categ_id][0]
                )  # Schedules of catastrophes and damages must me generated at the same time.
                self._inflict_peril(categ_id=categ_id, damage=damage_extent, t=t)
                self.rc_event_damage[categ_id] = self.rc_event_damage[categ_id][1:]
                # TODO: Ideally don't want to be taking from the beginning of lists, consider having soonest events at
                #  the end of the list. Probably fine though, only happens once per iteration
            else:
                if isleconfig.verbose:
                    print("Next peril ", self.rc_event_schedule[categ_id])

        # Shuffle risks (insurance and reinsurance risks)
        self._shuffle_risks()

        # Reset reinweights
        self._reset_reinsurance_weights()

        # Iterate reinsurnace firm agents
        for reinagent in self.reinsurancefirms:
            reinagent.iterate(t)

        # remove all non-accepted reinsurance risks
        self.reinrisks = []

        # Reset weights
        self._reset_insurance_weights()

        # Iterate insurance firm agents
        for agent in self.insurancefirms:
            agent.iterate(t)

        # Iterate catbonds
        for agent in self.catbonds:
            agent.iterate(t)

        self.insurance_models_counter = np.zeros(
            self.simulation_parameters["no_categories"]
        )

        # TODO: this and the next look like they could be cleaner
        for insurer in self.insurancefirms:
            if insurer.operational:
                for i in range(len(self.inaccuracy)):
                    if insurer.riskmodel.inaccuracy == self.inaccuracy[i]:
                        self.insurance_models_counter[i] += 1

        self.reinsurance_models_counter = np.zeros(
            self.simulation_parameters["no_categories"]
        )

        for reinsurer in self.reinsurancefirms:
            for i in range(len(self.inaccuracy)):
                if reinsurer.operational:
                    if reinsurer.riskmodel.inaccuracy == self.inaccuracy[i]:
                        self.reinsurance_models_counter[i] += 1

        network_division = 2  # How often network is updated.
        if isleconfig.show_network and t % network_division == 0 and t > 0:
            if t == network_division:
                self.RN = (
                    visualization_network.ReinsuranceNetwork()
                )  # Only creates once instance so only one figure.
            self.RN.update(self.insurancefirms, self.reinsurancefirms, self.catbonds)
            self.RN.visualize()

    def save_data(self):
        """Method to collect statistics about the current state of the simulation. Will pass these to the 
           Logger object (self.logger) to be recorded.
            No arguments.
            Returns None."""

        """ collect data """
        total_cash_no = sum(
            [insurancefirm.cash for insurancefirm in self.insurancefirms]
        )
        total_excess_capital = sum(
            [
                insurancefirm.get_excess_capital()
                for insurancefirm in self.insurancefirms
            ]
        )
        total_profitslosses = sum(
            [insurancefirm.get_profitslosses() for insurancefirm in self.insurancefirms]
        )
        total_contracts_no = sum(
            [
                len(insurancefirm.underwritten_contracts)
                for insurancefirm in self.insurancefirms
            ]
        )
        total_reincash_no = sum(
            [reinsurancefirm.cash for reinsurancefirm in self.reinsurancefirms]
        )
        total_reinexcess_capital = sum(
            [
                reinsurancefirm.get_excess_capital()
                for reinsurancefirm in self.reinsurancefirms
            ]
        )
        total_reinprofitslosses = sum(
            [
                reinsurancefirm.get_profitslosses()
                for reinsurancefirm in self.reinsurancefirms
            ]
        )
        total_reincontracts_no = sum(
            [
                len(reinsurancefirm.underwritten_contracts)
                for reinsurancefirm in self.reinsurancefirms
            ]
        )
        operational_no = sum(
            [insurancefirm.operational for insurancefirm in self.insurancefirms]
        )
        reinoperational_no = sum(
            [reinsurancefirm.operational for reinsurancefirm in self.reinsurancefirms]
        )
        catbondsoperational_no = sum([catbond.operational for catbond in self.catbonds])

        """ collect agent-level data """
        insurance_firms = [
            (insurancefirm.cash, insurancefirm.id, insurancefirm.operational)
            for insurancefirm in self.insurancefirms
        ]
        reinsurance_firms = [
            (reinsurancefirm.cash, reinsurancefirm.id, reinsurancefirm.operational)
            for reinsurancefirm in self.reinsurancefirms
        ]

        """ prepare dict """
        current_log = {}  # TODO: rewrite this as a single dictionary literal?
        current_log["total_cash"] = total_cash_no
        current_log["total_excess_capital"] = total_excess_capital
        current_log["total_profitslosses"] = total_profitslosses
        current_log["total_contracts"] = total_contracts_no
        current_log["total_operational"] = operational_no
        current_log["total_reincash"] = total_reincash_no
        current_log["total_reinexcess_capital"] = total_reinexcess_capital
        current_log["total_reinprofitslosses"] = total_reinprofitslosses
        current_log["total_reincontracts"] = total_reincontracts_no
        current_log["total_reinoperational"] = reinoperational_no
        current_log["total_catbondsoperational"] = catbondsoperational_no
        current_log["market_premium"] = self.market_premium
        current_log["market_reinpremium"] = self.reinsurance_market_premium
        current_log["cumulative_bankruptcies"] = self.cumulative_bankruptcies
        current_log["cumulative_market_exits"] = self.cumulative_market_exits
        current_log[
            "cumulative_unrecovered_claims"
        ] = self.cumulative_unrecovered_claims
        # Log the cumulative claims received so far.
        current_log["cumulative_claims"] = self.cumulative_claims

        """ add agent-level data to dict"""
        current_log["insurance_firms_cash"] = insurance_firms
        current_log["reinsurance_firms_cash"] = reinsurance_firms
        current_log["market_diffvar"] = self.compute_market_diffvar()

        current_log["individual_contracts"] = [
            len(insurancefirm.underwritten_contracts)
            for insurancefirm in self.insurancefirms
        ]

        """ call to Logger object """
        self.logger.record_data(current_log)

    # This function allows to return in a list all the data generated by the model. There is no other way to transfer
    # it back from the cloud.
    def obtain_log(self, requested_logs=None):
        return self.logger.obtain_log(requested_logs)

    def finalize(self, *args):
        """Function to handle operations after the end of the simulation run.
           Currently empty.
           It may be used to handle e.g. logging by including:
            self.log()
           but logging has been moved to start.py and ensemble.py
           """
        pass

    def _inflict_peril(self, categ_id, damage, t):
        """Method that calculates percentage damage done to each underwritten risk that is affected in the category
                    that event happened in. Passes values to allow calculation contracts to be resolved.
                    Arguments:
                        ID of category events took place
                        Given severity of damage from pareto distribution
                        Time iteration
                    No return value"""
        affected_contracts = [
            contract
            for insurer in self.insurancefirms
            for contract in insurer.underwritten_contracts
            if contract.category == categ_id
        ]
        if isleconfig.verbose:
            print("**** PERIL", damage)
        damagevalues = np.random.beta(
            1, 1.0 / damage - 1, size=self.risks_counter[categ_id]
        )
        uniformvalues = np.random.uniform(0, 1, size=self.risks_counter[categ_id])
        [
            contract.explode(t, uniformvalues[i], damagevalues[i])
            for i, contract in enumerate(affected_contracts)
        ]

    def receive_obligation(self, amount, recipient, due_time, purpose):
        """Method for adding obligation to list that is resolved at the start if each iteration of simulation. Only
                    called by metainsuranceorg for adding interest to cash.
                    Arguments
                        Amount: obligation value
                        Recipient: Who obligation is owed to
                        Due Time
                        Purpose: Reason for obligation (Interest due)
                    Returns None"""
        obligation = {
            "amount": amount,
            "recipient": recipient,
            "due_time": due_time,
            "purpose": purpose,
        }
        self.obligations.append(obligation)

    def _effect_payments(self, time):
        """Method for checking and paying obligation if due.
                    Arguments
                        Current time to allow check if due
                    Returns None"""
        if self.get_operational():
            due = [item for item in self.obligations if item["due_time"] <= time]
            self.obligations = [
                item for item in self.obligations if item["due_time"] > time
            ]
            # sum_due = sum([item["amount"] for item in due])
            for obligation in due:
                self._pay(obligation)

    def _pay(self, obligation):
        """Method for paying obligations called from effect_payments
            Accepts:
                Obligation: Type DataDict with categories amount, recipient, due time, purpose.
            Returns None"""
        amount = obligation["amount"]
        recipient = obligation["recipient"]
        purpose = obligation["purpose"]
        if not self.money_supply > amount:
            warnings.warn("Something wrong: economy out of money", RuntimeWarning)
        if recipient.get_operational():
            self.money_supply -= amount
            recipient.receive(amount)

    def receive(self, amount):
        """Method to accept cash payments. As insurance simulation cash is economy, adds money to total economy.
            Accepts:
                Amount due: Type Integer
            Returns None"""
        self.money_supply += amount

    def _reduce_money_supply(self, amount):
        """Method to reduce money supply immediately and without payment recipient (used to adjust money supply
         to compensate for agent endowment).
         Accepts:
                amount: Type Integer"""
        self.money_supply -= amount
        assert self.money_supply >= 0

    def _reset_reinsurance_weights(self):
        """Method for clearing and setting reinsurance weights dependant on how many reinsurance companies exist and
            how many offered reinsurance risks there are."""
        self.not_accepted_reinrisks = []

        operational_reinfirms = [
            reinsurancefirm
            for reinsurancefirm in self.reinsurancefirms
            if reinsurancefirm.operational
        ]

        operational_no = len(operational_reinfirms)

        reinrisks_no = len(self.reinrisks)

        self.reinsurers_weights = {}

        for reinsurer in self.reinsurancefirms:
            self.reinsurers_weights[reinsurer.id] = 0

        if operational_no > 0:

            if (
                reinrisks_no > operational_no
            ):  # QUERY: verify this - should all risk go to a reinsurer?
                weights = reinrisks_no / operational_no
                for reinsurer in self.reinsurancefirms:
                    self.reinsurers_weights[reinsurer.id] = math.floor(weights)
            else:
                for i in range(len(self.reinrisks)):
                    s = math.floor(np.random.uniform(0, len(operational_reinfirms), 1))
                    self.reinsurers_weights[operational_reinfirms[s].id] += 1
        else:
            self.not_accepted_reinrisks = self.reinrisks

    def _reset_insurance_weights(self):
        """Method for clearing and setting insurance weights dependant on how many insurance companies exist and
            how many insurance risks are offered. This determined which risks are sent to metainsuranceorg
            iteration."""
        operational_no = sum(
            [insurancefirm.operational for insurancefirm in self.insurancefirms]
        )

        operational_firms = [
            insurancefirm
            for insurancefirm in self.insurancefirms
            if insurancefirm.operational
        ]

        risks_no = len(self.risks)

        self.insurers_weights = {}

        for insurer in self.insurancefirms:
            self.insurers_weights[insurer.id] = 0

        if operational_no > 0:

            if risks_no > operational_no:  # TODO: as above
                weights = risks_no / operational_no
                for insurer in self.insurancefirms:
                    self.insurers_weights[insurer.id] = math.floor(weights)
            else:
                for i in range(len(self.risks)):
                    s = math.floor(np.random.uniform(0, len(operational_firms), 1))
                    self.insurers_weights[operational_firms[s].id] += 1

    def _shuffle_risks(self):
        """Method for shuffling risks."""
        np.random.shuffle(self.reinrisks)
        np.random.shuffle(self.risks)

    def _adjust_market_premium(self, capital):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the insurance market (insurance only).
               No return value.
           This method adjusts the premium charged by insurance firms for the risks covered. The premium reduces linearly
           with the capital available in the insurance market and viceversa. The premium reduces until it reaches a minimum
           below which no insurer is willing to reduce further the price. This method is only called in the self.iterate()
           method of this class."""
        self.market_premium = self.norm_premium * (
            self.simulation_parameters["upper_price_limit"]
            - self.simulation_parameters["premium_sensitivity"]
            * capital
            / (
                self.simulation_parameters["initial_agent_cash"]
                * self.damage_distribution.mean()
                * self.simulation_parameters["no_risks"]
            )
        )
        self.market_premium = max(
            self.market_premium,
            self.norm_premium * self.simulation_parameters["lower_price_limit"],
        )

    def _adjust_reinsurance_market_premium(self, capital):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the reinsurance market (reinsurance only).
               No return value.
           This method adjusts the premium charged by reinsurance firms for the risks covered. The premium reduces linearly
           with the capital available in the reinsurance market and viceversa. The premium reduces until it reaches a minimum
           below which no reinsurer is willing to reduce further the price. This method is only called in the self.iterate()
           method of this class."""
        self.reinsurance_market_premium = self.norm_premium * (
            self.simulation_parameters["upper_price_limit"]
            - self.simulation_parameters["reinpremium_sensitivity"]
            * capital
            / (
                self.simulation_parameters["initial_agent_cash"]
                * self.damage_distribution.mean()
                * self.simulation_parameters["no_risks"]
            )
        )
        self.reinsurance_market_premium = max(
            self.reinsurance_market_premium,
            self.norm_premium * self.simulation_parameters["lower_price_limit"],
        )

    def get_market_premium(self):
        """Get_market_premium Method.
               Accepts no arguments.
               Returns:
                   self.market_premium: Type float. The current insurance market premium.
           This method returns the current insurance market premium."""
        return self.market_premium

    def get_market_reinpremium(self):
        # QUERY: What's the difference between this and get_reinsurance_premium?
        """Get_market_reinpremium Method.
               Accepts no arguments.
               Returns:
                   self.reinsurance_market_premium: Type float. The current reinsurance market premium.
           This method returns the current reinsurance market premium."""
        return self.reinsurance_market_premium

    def get_reinsurance_premium(self, np_reinsurance_deductible_fraction):
        """Method to determine reinsurance premium based on deductible fraction
            Accepts:
                np_reinsurance_deductible_fraction: Type Integer
            Returns reinsurance premium (Type: Integer)"""
        # TODO: cut this out of the insurance market premium -> OBSOLETE??
        # TODO: make max_reduction into simulation_parameter ?
        if self.reinsurance_off:
            return float("inf")
        max_reduction = 0.1
        # QUERY: why is this this way? Why not, say, 1.0 - min(max_reduction * np_reinsurance_deductible_fraction)?
        return self.reinsurance_market_premium * (
            1.0 - max_reduction * np_reinsurance_deductible_fraction
        )

    def get_cat_bond_price(self, np_reinsurance_deductible_fraction):
        """Method to calculate and return catbond price. If catbonds are not desired will return infinity so no catbonds
            will be issued. Otherwise calculates based on reinsurance market premium, catbond premium, deductible fraction.
           Accepts:
                np_reinsurance_deductible_fraction: Type Integer
           Returns:
                Calculated catbond price."""
        # TODO: implement function dependent on total capital in cat bonds and on deductible ()
        # TODO: make max_reduction and max_cat_bond_surcharge into simulation_parameters ?
        if self.catbonds_off:
            return float("inf")
        max_reduction = 0.9
        max_cat_bond_surcharge = 0.5
        # QUERY: again, what does max_reduction represent?
        return self.reinsurance_market_premium * (
            1.0
            + max_cat_bond_surcharge
            - max_reduction * np_reinsurance_deductible_fraction
        )

    def append_reinrisks(self, item):
        """Method for appending reinrisks to simulation instance. Called from insurancefirm
                    Accepts: item (Type: List)"""
        if item:
            self.reinrisks.append(item)

    def remove_reinrisks(self, risko):
        if risko is not None:
            self.reinrisks.remove(risko)

    def get_reinrisks(self):
        """Method for shuffling reinsurance risks
            Returns: reinsurance risks"""
        np.random.shuffle(self.reinrisks)
        return self.reinrisks

    def solicit_insurance_requests(self, insurer_id, cash, insurer):
        """Method for determining which risks are to be assessed by firms based on insurer weights
                    Accepts:
                        id: Type integer
                        cash: Type Integer
                        insurer: Type firm metainsuranceorg instance
                    Returns:
                        risks_to_be_sent: Type List"""
        risks_to_be_sent = self.risks[: int(self.insurers_weights[insurer.id])]
        self.risks = self.risks[int(self.insurers_weights[insurer.id]) :]
        for risk in insurer.risks_kept:
            risks_to_be_sent.append(risk)

        # QUERY: what actually is InsuranceFirm.risks_kept?
        insurer.risks_kept = []

        np.random.shuffle(risks_to_be_sent)

        return risks_to_be_sent

    def solicit_reinsurance_requests(self, id, cash, reinsurer):
        """Method for determining which reinsurance risks are to be assessed by firms based on reinsurer weights
                           Accepts:
                               id: Type integer
                               cash: Type Integer
                               reinsurer: Type firm metainsuranceorg instance
                           Returns:
                               reinrisks_to_be_sent: Type List"""
        reinrisks_to_be_sent = self.reinrisks[
            : int(self.reinsurers_weights[reinsurer.id])
        ]
        self.reinrisks = self.reinrisks[int(self.reinsurers_weights[reinsurer.id]) :]

        for reinrisk in reinsurer.reinrisks_kept:
            reinrisks_to_be_sent.append(reinrisk)

        reinsurer.reinrisks_kept = []

        np.random.shuffle(reinrisks_to_be_sent)

        return reinrisks_to_be_sent

    def return_risks(self, not_accepted_risks):
        """Method for adding risks that were not deemed acceptable to underwrite back to list of uninsured risks
                    Accepts:
                        not_accepted_risks: Type List
                    No return value"""
        self.risks += not_accepted_risks

    def return_reinrisks(self, not_accepted_risks):
        """Method for adding reinsuracne risks that were not deemed acceptable to list of unaccepted reinsurance risks
                    Cleared every round and is never called so redundant?
                            Accepts:
                                not_accepted_risks: Type List
                            Returns None"""
        self.not_accepted_reinrisks += not_accepted_risks

    # QUERY: What does this represent?
    def _get_all_riskmodel_combinations(self, n, rm_factor):
        """Method  for calculating riskmodels for each category based on the risk model inaccuracy parameter, and is
                    used purely to assign inaccuracy. Currently all equal and overwritten immediately.
                    Accepts:
                        rm_factor: Type Integer = risk model inaccuracy parameter
                    Returns:
                        riskmodels: Type list"""
        riskmodels = []
        for i in range(self.simulation_parameters["no_categories"]):
            riskmodel_combination = rm_factor * np.ones(
                self.simulation_parameters["no_categories"]
            )
            riskmodel_combination[i] = 1 / rm_factor
            riskmodels.append(riskmodel_combination.tolist())
        return riskmodels

    def firm_enters_market(self, prob=-1, agent_type="InsuranceFirm"):
        """Method to determine if re/insurance firm enters the market based on set entry probabilities and a random
                            integer generated between 0, 1.
                            Accepts:
                                agent_type: Type String
                            Returns:
                                 True if firm can enter market
                                 False if firm cannot enter market"""
        # TODO: Do firms really enter the market randomly, with at most one in each timestep?
        if prob == -1:
            if agent_type == "InsuranceFirm":
                prob = self.simulation_parameters[
                    "insurance_firm_market_entry_probability"
                ]
            elif agent_type == "ReinsuranceFirm":
                prob = self.simulation_parameters[
                    "reinsurance_firm_market_entry_probability"
                ]
            else:
                raise ValueError(
                    f"Unknown agent type. Simulation requested to create agent of type {agent_type}"
                )
        return np.random.random() < prob

    def record_bankruptcy(self):
        """Record_bankruptcy Method.
               Accepts no arguments.
               No return value.
           This method is called when a firm files for bankruptcy. It is only called from the method dissolve() from the
           class metainsuranceorg.py after the dissolution of the firm."""
        self.cumulative_bankruptcies += 1

    def record_market_exit(self):
        """Record_market_exit Method.
               Accepts no arguments.
               No return value.
           This method is used to record the firms that leave the market due to underperforming conditions. It is only called
           from the method dissolve() from the class metainsuranceorg.py after the dissolution of the firm."""
        self.cumulative_market_exits += 1

    def record_unrecovered_claims(self, loss):
        """Method for recording unrecovered claims. If firm runs out of money it cannot pay more claims and so that
            money is lost and recorded using this method.
            Accepts:
                loss: Type integer, value of lost claim
            No return value"""
        self.cumulative_unrecovered_claims += loss

    def record_claims(self, claims):
        """This method records every claim made to insurers and reinsurers. It is called from both insurers and
            reinsurers (metainsuranceorg.py)."""
        self.cumulative_claims += claims

    def log(self):
        """Method to save the data of the simulation.
            No accepted values
            No return values
           Calls logger instance to save all the data of the simulation to a file, has to return if background run or
           not for replicating instances. This depends on parameters force_foreground and if the run is replicating
           or not."""
        self.logger.save_log(self.background_run)

    def compute_market_diffvar(self):
        """Method for calculating difference between number of all firms and the total value at risk. Used only in save
                    data when adding to the logger data dict."""
        totalina = sum(
            [
                firm.var_counter_per_risk
                for firm in self.insurancefirms
                if firm.operational
            ]
        )

        totalreal = len([firm for firm in self.insurancefirms if firm.operational])

        totalina += sum(
            [
                reinfirm.var_counter_per_risk
                for reinfirm in self.reinsurancefirms
                if reinfirm.operational
            ]
        )

        totalreal += len(
            [reinfirm for reinfirm in self.reinsurancefirms if reinfirm.operational]
        )

        totaldiff = totalina - totalreal

        return totaldiff
        # self.history_logs['market_diffvar'].append(totaldiff)

    def get_unique_insurer_id(self):
        """Method for getting unique id for insurer. Used in initialising agents in start.py and insurancesimulation.
            Iterates after each call so id is unique to each firm.
           Returns:
                current_id: Type integer"""
        current_id = self.insurer_id_counter
        self.insurer_id_counter += 1
        return current_id

    def get_unique_reinsurer_id(self):
        """Method for getting unique id for insurer. Used in initialising agents in start.py and insurancesimulation.
            Iterates after each call so id is unique to each firm.
           Returns:
                current_id: Type integer"""
        current_id = self.reinsurer_id_counter
        self.reinsurer_id_counter += 1
        return current_id

    def insurance_entry_index(self):
        """Method that returns the entry index for insurance firms, i.e. the index for the initial agent parameters
                   that is taken from the list of already created parameters.
               Returns:
                   Indices of the type of riskmodel that the least firms are using."""
        return self.insurance_models_counter[
            0 : self.simulation_parameters["no_riskmodels"]
        ].argmin()

    def reinsurance_entry_index(self):
        """Method that returns the entry index for reinsurance firms, i.e. the index for the initial agent parameters
                    that is taken from the list of already created parameters.
                Returns:
                    Indices of the type of riskmodel that the least reinsurance firms are using."""
        return self.reinsurance_models_counter[
            0 : self.simulation_parameters["no_riskmodels"]
        ].argmin()

    def get_operational(self):
        """Method to return if simulation is operational. Always true. Used only in pay methods above and
                    metainsuranceorg.
                   Accepts no arguments
                   Returns True"""
        return True

    def reinsurance_capital_entry(self):
        # This method determines the capital market entry (initial cash) of reinsurers. It is only run in start.py.
        capital_per_non_re_cat = []

        for reinrisk in self.not_accepted_reinrisks:
            capital_per_non_re_cat.append(reinrisk["value"])
        # It takes all the values of the reinsurance risks NOT REINSURED.

        # If there are any non-reinsured risks going, take a sample of them and have starting capital equal to twice
        # the maximum value among that sample.  # QUERY: why this particular value?
        if len(capital_per_non_re_cat) > 0:
            # We only perform this action if there are reinsurance contracts that have
            # not been reinsured in the last time period.
            capital_per_non_re_cat = np.random.choice(
                capital_per_non_re_cat, 10
            )  # Only 10 values sampled randomly are considered. (Too low?)
            entry = max(
                capital_per_non_re_cat
            )  # For market entry the maximum of the sample is considered.
            entry = (
                2 * entry
            )  # The capital market entry of those values will be the double of the maximum.
        else:  # Otherwise the default reinsurance cash market entry is considered.
            entry = self.simulation_parameters["initial_reinagent_cash"]

        return entry  # The capital market entry is returned.

    def reset_pls(self):
        """Reset_pls Method.
               Accepts no arguments.
               No return value.
           This method reset all the profits and losses of all insurance firms, reinsurance firms and catbonds."""
        for insurancefirm in self.insurancefirms:
            insurancefirm.reset_pl()

        for reininsurancefirm in self.reinsurancefirms:
            reininsurancefirm.reset_pl()

        for catbond in self.catbonds:
            catbond.reset_pl()
