import math
import random
import copy
import logger
import warnings

import scipy.stats
import numpy as np

from distributiontruncated import TruncatedDistWrapper
import insurancefirms
from centralbank import CentralBank
import isleconfig
from genericclasses import GenericAgent, RiskProperties, AgentProperties, Constant
import catbond

from typing import (
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Any,
    Optional,
    Collection,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genericclasses import Distribution
    from metainsuranceorg import MetaInsuranceOrg

if isleconfig.show_network or isleconfig.save_network:
    import visualization_network


class InsuranceSimulation(GenericAgent):
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
        override_no_riskmodels: bool,
        replic_id: int,
        simulation_parameters: MutableMapping,
        rc_event_schedule: MutableSequence[MutableSequence[int]],
        rc_event_damage: MutableSequence[MutableSequence[float]],
        damage_distribution: "Distribution" = TruncatedDistWrapper(
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
        super().__init__()
        "Override one-riskmodel case (this is to ensure all other parameters are truly identical for comparison runs)"
        if override_no_riskmodels:
            simulation_parameters["no_riskmodels"] = override_no_riskmodels
        self.number_riskmodels: int = simulation_parameters["no_riskmodels"]

        "Save parameters, sets parameters of sim according to isleconfig.py"
        if (replic_id is None) or isleconfig.force_foreground:
            self.background_run = False
        else:
            self.background_run = True
        self.replic_id = replic_id
        self.simulation_parameters: MutableMapping = simulation_parameters
        self.simulation_parameters["simulation"] = self

        "Unpacks parameters and sets distributions"
        self.damage_distribution: "Distribution" = damage_distribution

        self.catbonds_off: bool = simulation_parameters["catbonds_off"]
        self.reinsurance_off: bool = simulation_parameters["reinsurance_off"]
        # TODO: It's actually geometric (in effect) - change?
        self.cat_separation_distribution = scipy.stats.expon(
            0, simulation_parameters["event_time_mean_separation"]
        )

        # Risk factors represent, for example, the earthquake risk for a particular house (compare to the value)
        # TODO: Implement! Think about insurers rejecting risks under certain situations (high risk factor)
        self.risk_factor_lower_bound: float = simulation_parameters[
            "risk_factor_lower_bound"
        ]
        self.risk_factor_spread: float = (
            simulation_parameters["risk_factor_upper_bound"]
            - self.risk_factor_lower_bound
        )
        if simulation_parameters["risk_factors_present"]:
            self.risk_factor_distribution = scipy.stats.uniform(
                loc=self.risk_factor_lower_bound, scale=self.risk_factor_spread
            )
        else:
            self.risk_factor_distribution = Constant(loc=1.0)
        self.risk_value_distribution = Constant(
            loc=simulation_parameters["value_per_risk"]
        )

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

        # norm_premium is the basic per-risk premium (for a contract of average length) charged to cover expected
        # losses and underwriting costs
        self.norm_premium: float = (
            expected_damage_frequency
            * self.damage_distribution.mean()
            * risk_factor_mean
            * (1 + self.simulation_parameters["norm_profit_markup"])
        )

        self.market_premium: float = self.norm_premium
        self.reinsurance_market_premium: float = self.market_premium
        # TODO: is this problematic as initial value? (later it is recomputed in every iteration)

        self.total_no_risks: int = simulation_parameters["no_risks"]

        "Set up monetary system (should instead be with the customers, if customers are modeled explicitly)"
        self.cash: float = self.simulation_parameters["money_supply"]
        self.bank = CentralBank(self.cash)

        "set up risk categories"
        self.riskcategories: Sequence[int] = list(
            range(self.simulation_parameters["no_categories"])
        )
        self.rc_event_schedule: MutableSequence[int] = []
        self.rc_event_damage: MutableSequence[float] = []

        # For debugging (cloud debugging) purposes is good to store the initial schedule of catastrophes
        # and damages that will be use in a single run of the model.
        self.rc_event_schedule_initial: Sequence[float] = []
        self.rc_event_damage_initial: Sequence[float] = []
        if (
            rc_event_schedule is not None and rc_event_damage is not None
        ):  # If we have schedules pass as arguments we used them.
            self.rc_event_schedule = copy.deepcopy(rc_event_schedule)
            self.rc_event_schedule_initial = copy.deepcopy(rc_event_schedule)

            self.rc_event_damage = copy.deepcopy(rc_event_damage)
            self.rc_event_damage_initial = copy.deepcopy(rc_event_damage)
        else:  # Otherwise the schedules and damages are generated.
            raise Exception("No event schedules and damages supplied")

        """Set up risks"""
        risk_value_mean = self.risk_value_distribution.mean()

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

        self.risks: Collection[RiskProperties] = [
            RiskProperties(
                risk_factor=rrisk_factors[i],
                value=rvalues[i],
                category=rcategories[i],
                owner=self,
            )
            for i in range(self.simulation_parameters["no_risks"])
        ]

        self.risks_counter: MutableSequence[int] = [
            0 for _ in range(self.simulation_parameters["no_categories"])
        ]

        for risk in self.risks:
            self.risks_counter[risk.category] += 1

        self.inaccuracy: Sequence[Sequence[int]] = self._get_all_riskmodel_combinations(
            self.simulation_parameters["riskmodel_inaccuracy_parameter"]
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
        self.agent_parameters: Mapping[str, MutableSequence[AgentProperties]] = {
            "insurancefirm": [],
            "reinsurancefirm": [],
        }
        self.insurer_id_counter: int = 0
        self.reinsurer_id_counter: int = 0
        self.catbond_id_counter: int = 0

        self.initialize_agent_parameters(
            "insurancefirm", simulation_parameters, risk_model_configurations
        )
        self.initialize_agent_parameters(
            "reinsurancefirm", simulation_parameters, risk_model_configurations
        )

        "Agent lists"
        self.reinsurancefirms: Collection = []
        self.insurancefirms: Collection = []
        self.catbonds: list = []

        "Lists of agent weights"
        self.insurers_weights: MutableMapping[int, float] = {}
        self.reinsurers_weights: MutableMapping[int, float] = {}

        "List of reinsurance risks offered for underwriting"
        self.reinrisks: Collection[RiskProperties] = []
        self.not_accepted_reinrisks: Collection[RiskProperties] = []

        "Cumulative variables for history and logging"
        self.cumulative_bankruptcies: int = 0
        self.cumulative_market_exits: int = 0
        self.cumulative_bought_firms: int = 0
        self.cumulative_nonregulation_firms: int = 0
        self.cumulative_unrecovered_claims: float = 0.0
        self.cumulative_claims: float = 0.0

        "Lists for firms that are to be sold."
        self.selling_insurance_firms = []
        self.selling_reinsurance_firms = []

        "Lists for logging history"
        self.logger: logger.Logger = logger.Logger(
            no_riskmodels=simulation_parameters["no_riskmodels"],
            rc_event_schedule_initial=self.rc_event_schedule_initial,
            rc_event_damage_initial=self.rc_event_damage_initial,
        )

        self.insurance_models_counter: np.ndarray = np.zeros(
            self.simulation_parameters["no_categories"]
        )
        self.reinsurance_models_counter: np.ndarray = np.zeros(
            self.simulation_parameters["no_categories"]
        )
        "Add initial set of agents"
        self.add_agents(
            insurancefirms.InsuranceFirm,
            "insurancefirm",
            n=self.simulation_parameters["no_insurancefirms"],
        )
        self.add_agents(
            insurancefirms.ReinsuranceFirm,
            "reinsurancefirm",
            n=self.simulation_parameters["no_reinsurancefirms"],
        )

        self._time: Optional[int] = None
        self.RN: Optional[visualization_network.ReinsuranceNetwork] = None

    def initialize_agent_parameters(
        self,
        firmtype: str,
        simulation_parameters: Mapping[str, Any],
        risk_model_configurations: Sequence[Mapping],
    ):
        """General function for initialising the agent parameters
            Takes the firm type as argument, also needing sim params and risk configs
            Creates the agent parameters of both firm types for the initial number specified in isleconfig.py
                Returns None"""
        if firmtype == "insurancefirm":
            no_firms = simulation_parameters["no_insurancefirms"]
            initial_cash = "initial_agent_cash"
            reinsurance_level_lowerbound = simulation_parameters[
                "insurance_reinsurance_levels_lower_bound"
            ]
            reinsurance_level_upperbound = simulation_parameters[
                "insurance_reinsurance_levels_upper_bound"
            ]

        elif firmtype == "reinsurancefirm":
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
            if firmtype == "insurancefirm":
                unique_id = self.get_unique_insurer_id()
            elif firmtype == "reinsurancefirm":
                unique_id = self.get_unique_reinsurer_id()
            else:
                raise ValueError(f"Firm type {firmtype} not recognised")

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
                AgentProperties(
                    id=unique_id,
                    initial_cash=simulation_parameters[initial_cash],
                    riskmodel_config=riskmodel_config,
                    norm_premium=self.norm_premium,
                    profit_target=simulation_parameters["norm_profit_markup"],
                    initial_acceptance_threshold=simulation_parameters[
                        "initial_acceptance_threshold"
                    ],
                    acceptance_threshold_friction=simulation_parameters[
                        "acceptance_threshold_friction"
                    ],
                    reinsurance_limit=simulation_parameters["reinsurance_limit"],
                    non_proportional_reinsurance_level=reinsurance_level,
                    capacity_target_decrement_threshold=simulation_parameters[
                        "capacity_target_decrement_threshold"
                    ],
                    capacity_target_increment_threshold=simulation_parameters[
                        "capacity_target_increment_threshold"
                    ],
                    capacity_target_decrement_factor=simulation_parameters[
                        "capacity_target_decrement_factor"
                    ],
                    capacity_target_increment_factor=simulation_parameters[
                        "capacity_target_increment_factor"
                    ],
                    interest_rate=simulation_parameters["interest_rate"],
                )
            )

    def add_agents(
        self,
        agent_class: type,
        agent_class_string: str,
        agents: "Sequence[GenericAgent]" = None,
        n: int = 1,
    ):
        """Method for building agents and adding them to the simulation. Can also add pre-made catbond agents directly
        Accepts:
            agent_class: class of the agent, InsuranceFirm, ReinsuranceFirm or CatBond
            agent_class_string: string of the same, "insurancefirm", "reinsurancefirm" or "catbond"
            agents: if adding directly, a list of the agents to add
            n: int of number of agents to add
        Returns:
            None"""
        if agents:
            # We're probably just adding a catbond
            if agent_class_string == "catbond":
                assert len(agents) == n
                self.catbonds += agents
            else:
                raise ValueError("Only catbonds may be passed directly")
        else:
            # We need to create and input the agents
            if agent_class_string == "insurancefirm":
                if not self.insurancefirms:
                    # There aren't any other firms yet, add the first ones
                    assert len(self.agent_parameters["insurancefirm"]) == n
                    agent_parameters = self.agent_parameters["insurancefirm"]
                else:
                    # We are adding new agents to an existing simulation
                    agent_parameters = [
                        self.agent_parameters["insurancefirm"][
                            self.insurance_entry_index()
                        ]
                        for _ in range(n)
                    ]
                    for ap in agent_parameters:
                        ap.id = self.get_unique_insurer_id()
                agents = [
                    agent_class(self.simulation_parameters, ap)
                    for ap in agent_parameters
                ]
                # We've made the agents, add them to the simulation
                self.insurancefirms += agents
                for _ in agents:
                    self.logger.add_firm("insurance")

            elif agent_class_string == "reinsurancefirm":
                # Much the same as above
                if not self.reinsurancefirms:
                    assert len(self.agent_parameters["reinsurancefirm"]) == n
                    agent_parameters = self.agent_parameters["reinsurancefirm"]
                else:
                    agent_parameters = [
                        self.agent_parameters["reinsurancefirm"][
                            self.reinsurance_entry_index()
                        ]
                        for _ in range(n)
                    ]
                    for ap in agent_parameters:
                        ap.id = self.get_unique_reinsurer_id()
                        # QUERY: This was written but not actually used in the original implementation - should it be?
                        # ap.initial_cash = self.reinsurance_capital_entry()
                agents = [
                    agent_class(self.simulation_parameters, ap)
                    for ap in agent_parameters
                ]
                self.reinsurancefirms += agents
                for _ in agents:
                    self.logger.add_firm("reinsurance")

            elif agent_class_string == "catbond":
                raise ValueError(f"Catbonds must be built before being added")
            else:
                raise ValueError(f"Unrecognised agent type {agent_class_string}")

            # Keep the total amount of money constant
            total_new_agent_cash = sum([agent.cash for agent in agents])
            self._reduce_money_supply(total_new_agent_cash)

    def delete_agents(self, agents: Sequence[catbond.CatBond]):
        """Method for deleting catbonds as it is only agent that is allowed to be removed
            alters lists of catbonds
              Returns none"""
        for agent in agents:
            if isinstance(agent, catbond.CatBond):
                self.catbonds.remove(agent)
            else:
                raise ValueError(
                    f"Trying to remove unremovable agent, type: {type(agent)}"
                )

    def iterate(self, t: int):
        """Function that is called from start.py for each iteration that settles obligations, capital then reselects
            risks for the insurance and reinsurance companies to evaluate. Firms are then iterated through to accept
              new risks, _pay obligations, increase capacity etc.
           Accepts:
                t: Integer, current time step
           Returns None"""

        self._time = t
        if isleconfig.verbose:
            print()
            print(t, ": ", len(self.risks))
        if isleconfig.showprogress:
            print(f"\rTime: {t}", end="")

        if self.firm_enters_market(agent_type="InsuranceFirm"):
            self.add_agents(insurancefirms.InsuranceFirm, "insurancefirm", n=1)

        if self.firm_enters_market(agent_type="ReinsuranceFirm"):
            self.add_agents(insurancefirms.ReinsuranceFirm, "reinsurancefirm", n=1)

        self.reset_pls()

        # adjust market premiums
        sum_capital = sum([agent.get_cash() for agent in self.insurancefirms])
        self._adjust_market_premium(capital=sum_capital)
        sum_capital = sum([agent.get_cash() for agent in self.reinsurancefirms])
        self._adjust_reinsurance_market_premium(capital=sum_capital)

        # Pay obligations
        self._effect_payments(t)

        # identify perils and effect claims
        damage_extent = None
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
                # Schedules of catastrophes and damages must me generated at the same time.
                self.rc_event_schedule[categ_id] = self.rc_event_schedule[categ_id][1:]
                damage_extent = copy.copy(self.rc_event_damage[categ_id][0])
                self._inflict_peril(categ_id=categ_id, damage=damage_extent, t=t)
                del self.rc_event_damage[categ_id][0]

            else:
                if isleconfig.verbose:
                    print("Next peril ", self.rc_event_schedule[categ_id])

        # Provide government aid if damage severe enough
        if self.simulation_parameters["aid_relief"]:
            self.bank.adjust_aid_budget(time=t)
            if damage_extent is not None:
                op_firms = [firm for firm in self.insurancefirms if firm.operational]
                aid_dict = self.bank.provide_aid(op_firms, damage_extent, time=t)
                for key in aid_dict.keys():
                    self.receive_obligation(
                        amount=aid_dict[key], recipient=key, due_time=t, purpose="aid"
                    )

        # Shuffle risks (insurance and reinsurance risks)
        self._shuffle_risks()

        # Reset reinweights
        self._reset_reinsurance_weights()

        # Iterate reinsurnace firm agents
        for reinagent in self.reinsurancefirms:
            if reinagent.operational:
                reinagent.iterate(t)

                if reinagent.cash < 0:
                    print(f"Reinsurer {reinagent.id} has negative cash")

        if self.simulation_parameters["buy_bankruptcies"]:
            for reinagent in self.reinsurancefirms:
                if reinagent.operational:
                    reinagent.consider_buyout(firm_type="reinsurer")

        # remove all non-accepted reinsurance risks
        self.reinrisks = []

        # Reset weights
        self._reset_insurance_weights()

        # Iterate insurance firm agents
        for agent in self.insurancefirms:
            if agent.operational:
                agent.iterate(t)

                if agent.cash < 0:
                    print(f"Insurer {agent.id} has negative cash")

        if self.simulation_parameters["buy_bankruptcies"]:
            for agent in self.insurancefirms:
                if agent.operational:
                    agent.consider_buyout(firm_type="insurer")

        # Reset list of bankrupt insurance firms
        self.reset_selling_firms()

        # Iterate catbonds
        for agent in self.catbonds.copy():
            # If we don't take the copy we have a *very* bad time with some *very* frustrating debugging, as catbonds
            # can delete themselves from self.catbonds, which causes some catbonds to not be iterated(!)
            agent.iterate(t)

        self.insurance_models_counter = np.zeros(
            self.simulation_parameters["no_categories"]
        )

        self._update_model_counters()

        """
        network_division = 1  # How often network is updated.
        if (
            (isleconfig.show_network or isleconfig.save_network)
            and t % network_division == 0
            and t > 0
        ):
            if t == network_division:  # Only creates once instance so only one figure.
                self.RN = visualization_network.ReinsuranceNetwork(
                    self.rc_event_schedule_initial
                )

            self.RN.update(self.insurancefirms, self.reinsurancefirms, self.catbonds)

            if isleconfig.show_network:
                self.RN.visualize()
            if isleconfig.save_network and t == (
                self.simulation_parameters["max_time"] - 800
            ):
                self.RN.save_network_data()
                print("Network data has been saved to data/network_data.dat")
            """

    def save_data(self):
        """Method to collect statistics about the current state of the simulation. Will pass these to the
           Logger object (self.logger) to be recorded.
            No arguments.
            Returns None."""

        """ collect data """
        total_cash_no = sum([firm.cash for firm in self.insurancefirms])
        total_excess_capital = sum(
            [firm.get_excess_capital() for firm in self.insurancefirms]
        )
        total_profitslosses = sum(
            [firm.get_profitslosses() for firm in self.insurancefirms]
        )
        total_contracts_no = sum(
            [len(firm.underwritten_contracts) for firm in self.insurancefirms]
        )
        total_reincash_no = sum([firm.cash for firm in self.reinsurancefirms])
        total_reinexcess_capital = sum(
            [firm.get_excess_capital() for firm in self.reinsurancefirms]
        )
        total_reinprofitslosses = sum(
            [firm.get_profitslosses() for firm in self.reinsurancefirms]
        )
        total_reincontracts_no = sum(
            [len(firm.underwritten_contracts) for firm in self.reinsurancefirms]
        )
        operational_no = sum([firm.operational for firm in self.insurancefirms])
        reinoperational_no = sum([firm.operational for firm in self.reinsurancefirms])
        catbondsoperational_no = sum([cb.operational for cb in self.catbonds])

        """ collect agent-level data """
        insurance_firms = [firm.cash for firm in self.insurancefirms]
        reinsurance_firms = [firm.cash for firm in self.reinsurancefirms]
        insurance_contracts = [
            len(firm.underwritten_contracts) for firm in self.insurancefirms
        ]
        reinsurance_contracts = [
            len(firm.underwritten_contracts) for firm in self.reinsurancefirms
        ]
        ins_dividends = sum([firm.dividends_paid for firm in self.insurancefirms])
        re_dividends = sum([firm.dividends_paid for firm in self.reinsurancefirms])

        """ prepare dict """
        current_log = {
            "total_cash": total_cash_no,
            "total_excess_capital": total_excess_capital,
            "total_profitslosses": total_profitslosses,
            "total_contracts": total_contracts_no,
            "total_operational": operational_no,
            "total_reincash": total_reincash_no,
            "total_reinexcess_capital": total_reinexcess_capital,
            "total_reinprofitslosses": total_reinprofitslosses,
            "total_reincontracts": total_reincontracts_no,
            "total_reinoperational": reinoperational_no,
            "total_catbondsoperational": catbondsoperational_no,
            "market_premium": self.market_premium,
            "market_reinpremium": self.reinsurance_market_premium,
            "cumulative_bankruptcies": self.cumulative_bankruptcies,
            "cumulative_market_exits": self.cumulative_market_exits,
            "cumulative_unrecovered_claims": self.cumulative_unrecovered_claims,
            "cumulative_claims": self.cumulative_claims,
            "cumulative_bought_firms": self.cumulative_bought_firms,
            "cumulative_nonregulation_firms": self.cumulative_nonregulation_firms,
            "insurance_firms_cash": insurance_firms,
            "reinsurance_firms_cash": reinsurance_firms,
            "market_diffvar": self.compute_market_diffvar(),
            "individual_contracts": insurance_contracts,
            "reinsurance_contracts": reinsurance_contracts,
            "insurance_cumulative_dividends": ins_dividends,
            "reinsurance_cumulative_dividends": re_dividends,
        }

        if isleconfig.save_network:
            adj_list, node_labels, edge_labels, num_entities = (
                self.update_network_data()
            )
            current_log["unweighted_network_data"] = adj_list
            current_log["network_node_labels"] = node_labels
            current_log["network_edge_labels"] = edge_labels
            current_log["number_of_agents"] = num_entities

        """ call to Logger object """
        self.logger.record_data(current_log)

    def obtain_log(self, requested_logs: Mapping = None) -> MutableSequence:
        """This function allows to return in a list all the data generated by the model. There is no other way to
            transfer it back from the cloud."""
        return self.logger.obtain_log(requested_logs)

    def _inflict_peril(self, categ_id: int, damage: float, t: int):
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
            a=1, b=1.0 / damage - 1, size=len(affected_contracts)
        )
        uniformvalues = np.random.uniform(0, 1, size=len(affected_contracts))
        for i, contract in enumerate(affected_contracts):
            contract.explode(t, uniformvalues[i], damagevalues[i])

    def enter_illiquidity(self, time: int, sum_due: float):
        raise RuntimeError("Oh no, economy has run out of money!")

    def _reduce_money_supply(self, amount: float):
        """Method to reduce money supply immediately and without payment recipient (used to adjust money supply
         to compensate for agent endowment).
         Accepts:
                amount: Type Integer"""
        self.cash -= amount
        self.bank.update_money_supply(amount, reduce=True)
        assert self.cash >= 0

    def _reset_reinsurance_weights(self):
        """Method for clearing and setting reinsurance weights dependant on how many reinsurance companies exist and
            how many offered reinsurance risks there are."""
        self.not_accepted_reinrisks = []

        operational_reinfirms = [
            firm for firm in self.reinsurancefirms if firm.operational
        ]

        operational_no = len(operational_reinfirms)

        reinrisks_no = len(self.reinrisks)

        self.reinsurers_weights = {}

        for reinsurer in self.reinsurancefirms:
            self.reinsurers_weights[reinsurer.id] = 0

        if operational_no > 0:

            if reinrisks_no > operational_no:
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
        operational_no = sum([firm.operational for firm in self.insurancefirms])

        operational_firms = [firm for firm in self.insurancefirms if firm.operational]

        risks_no = len(self.risks)

        self.insurers_weights = {}

        for insurer in self.insurancefirms:
            self.insurers_weights[insurer.id] = 0

        if operational_no > 0:

            if risks_no > operational_no:
                weights = risks_no / operational_no
                for insurer in self.insurancefirms:
                    self.insurers_weights[insurer.id] = math.floor(weights)
            else:
                for i in range(len(self.risks)):
                    s = math.floor(np.random.uniform(0, len(operational_firms), 1))
                    self.insurers_weights[operational_firms[s].id] += 1

    def _update_model_counters(self):
        for insurer in self.insurancefirms:
            if insurer.operational:
                for i in range(len(self.inaccuracy)):
                    if np.array_equal(insurer.riskmodel.inaccuracy, self.inaccuracy[i]):
                        self.insurance_models_counter[i] += 1

        self.reinsurance_models_counter = np.zeros(
            self.simulation_parameters["no_categories"]
        )

        for reinsurer in self.reinsurancefirms:
            for i in range(len(self.inaccuracy)):
                if reinsurer.operational:
                    if np.array_equal(
                        reinsurer.riskmodel.inaccuracy, self.inaccuracy[i]
                    ):
                        self.reinsurance_models_counter[i] += 1

    def _shuffle_risks(self):
        """Method for shuffling risks."""
        np.random.shuffle(self.reinrisks)
        np.random.shuffle(self.risks)

    def _adjust_market_premium(self, capital: float):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the insurance market (insurance only).
               No return value.
           This method adjusts the premium charged by insurance firms for the risks covered. The premium reduces
           linearly with the capital available in the insurance market and viceversa. The premium reduces until it
           reaches a minimum below which no insurer is willing to reduce further the price. This method is only called
           in the self.iterate() method of this class."""
        # QUERY: Why is initial_agent_cash used here?
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

    def _adjust_reinsurance_market_premium(self, capital: float):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the reinsurance market (reinsurance only).
               No return value.
           This method adjusts the premium charged by reinsurance firms for the risks covered. The premium reduces
           linearly with the capital available in the reinsurance market and viceversa. The premium reduces until it
           reaches a minimum below which no reinsurer is willing to reduce further the price. This method is only
           called in the self.iterate() method of this class."""
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

    def get_market_premium(self) -> float:
        """Get_market_premium Method.
               Accepts no arguments.
               Returns:
                   self.market_premium: Type float. The current insurance market premium.
           This method returns the current insurance market premium."""
        return self.market_premium

    def get_market_reinpremium(self) -> float:
        # QUERY: What's the difference between this and get_reinsurance_premium below?
        """Get_market_reinpremium Method.
               Accepts no arguments.
               Returns:
                   self.reinsurance_market_premium: Type float. The current reinsurance market premium.
           This method returns the current reinsurance market premium."""
        return self.reinsurance_market_premium

    def get_reinsurance_premium(
        self, deductible_fraction: float, limit_fraction: float = 1
    ) -> float:
        """Method to determine reinsurance premium based on deductible fraction
            Accepts:
                np_reinsurance_deductible_fraction: Type Integer
            Returns reinsurance premium (Type: Integer)"""
        # TODO: make max_reduction into simulation_parameter ?
        if self.reinsurance_off:
            return float("inf")
        else:
            max_reduction = 0.1
            return self.reinsurance_market_premium * (
                1.0 - max_reduction * (1 - limit_fraction + deductible_fraction)
            )

    def get_cat_bond_price(
        self, deductible_fraction: float, excess_fraction: float = 1
    ) -> float:
        """Method to calculate and return catbond price. If catbonds are not desired will return infinity so no
            catbonds will be issued. Otherwise calculates based on reinsurance market premium, catbond premium,
            deductible fraction.
           Accepts:
                np_reinsurance_deductible_fraction: Type Integer
           Returns:
                Calculated catbond price."""
        # TODO: make max_reduction and max_cat_bond_surcharge into simulation_parameters ?
        if self.catbonds_off:
            return float("inf")
        max_reduction = 0.9
        max_cat_bond_surcharge = 0.5
        # QUERY: again, what does max_reduction represent?
        # TODO: How should this relate to deductible and excess?
        return self.reinsurance_market_premium * (
            1.0
            + max_cat_bond_surcharge
            - max_reduction * (deductible_fraction + 1 - excess_fraction)
        )

    def append_reinrisks(self, reinrisk: RiskProperties):
        """Method for appending reinrisks to simulation instance. Called from insurancefirm
                    Accepts: item (Type: List)"""
        if reinrisk:
            self.reinrisks.append(reinrisk)

    def remove_reinrisks(
        self, risko: RiskProperties = None, firm: "MetaInsuranceOrg" = None
    ):
        """Either removes a single reinrisk or all reinrisks requested by a given firm (probably because it has gone
        under)"""
        if risko is not None:
            self.reinrisks.remove(risko)
        elif firm is not None:
            self.reinrisks = [risk for risk in self.reinrisks if risk.owner is not firm]

    def get_reinrisks(self) -> Collection[RiskProperties]:
        """Method for shuffling reinsurance risks
            Returns: reinsurance risks"""
        np.random.shuffle(self.reinrisks)
        return self.reinrisks

    def solicit_insurance_requests(
        self, insurer: "MetaInsuranceOrg"
    ) -> Sequence[RiskProperties]:
        """Method for determining which risks are to be assessed by firms based on insurer weights
                    Accepts:
                        insurer: Type firm metainsuranceorg instance
                    Returns:
                        risks_to_be_sent: Type List"""
        risks_to_be_sent = self.risks[: int(self.insurers_weights[insurer.id])]
        self.risks = self.risks[int(self.insurers_weights[insurer.id]) :]
        for risk in insurer.risks_retained:
            risks_to_be_sent.append(risk)

        insurer.risks_retained = []

        np.random.shuffle(risks_to_be_sent)

        return risks_to_be_sent

    def solicit_reinsurance_requests(
        self, reinsurer: "MetaInsuranceOrg"
    ) -> Sequence[RiskProperties]:
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

        for reinrisk in reinsurer.reinrisks_retained:
            if reinrisk.owner.operational:
                reinrisks_to_be_sent.append(reinrisk)

        reinsurer.reinrisks_retained = []

        np.random.shuffle(reinrisks_to_be_sent)

        return reinrisks_to_be_sent

    def return_risks(self, not_accepted_risks: Sequence[RiskProperties]):
        """Method for adding risks that were not deemed acceptable to underwrite back to list of uninsured risks
                    Accepts:
                        not_accepted_risks: Type List
                    No return value"""
        self.risks += not_accepted_risks

    def return_reinrisks(self, not_accepted_risks: Sequence[RiskProperties]):
        """Method for adding reinsuracne risks that were not deemed acceptable to list of unaccepted reinsurance risks
                    Cleared every round and is never called so redundant?
                            Accepts:
                                not_accepted_risks: Type List
                            Returns None"""
        self.not_accepted_reinrisks += not_accepted_risks

    def _get_all_riskmodel_combinations(
        self, rm_factor: float
    ) -> Sequence[Sequence[float]]:
        """Method  for calculating riskmodels for each category based on the risk model inaccuracy parameter, and is
                    used purely to assign inaccuracy. Undervalues one risk category and overestimates all the rest.
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
            riskmodels.append(riskmodel_combination)
        return riskmodels

    def firm_enters_market(
        self, prob: float = -1, agent_type: str = "InsuranceFirm"
    ) -> bool:
        """Method to determine if re/insurance firm enters the market based on set entry probabilities and a random
                            integer generated between 0, 1.
                            Accepts:
                                agent_type: Type String
                            Returns:
                                 True if firm can enter market
                                 False if firm cannot enter market"""
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
                raise ValueError(f"Unknown agent type {agent_type}")
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
           This method is used to record the firms that leave the market due to underperforming conditions. It is
           only called from the method dissolve() from the class metainsuranceorg.py after the dissolution of the
           firm."""
        self.cumulative_market_exits += 1

    def record_nonregulation_firm(self):
        """Method to record non-regulation firm exits..
            Accepts no arguments.
            No return value.
        This method is used to record the firms that leave the market due to the regulator. It is
        only called from the method dissolve() from the class metainsuranceorg.py after the dissolution of the
        firm, and only if the regulator is working."""
        self.cumulative_nonregulation_firms += 1

    def record_bought_firm(self):
        """Method to record a firm bought.
            Accepts no arguments.
            No return value.
        This method is used to record the number of firms that have been bought. Only called from buyout() in
        metainsuranceorg.py after all obligations and contracts have been transferred to buyer."""
        self.cumulative_bought_firms += 1

    def record_unrecovered_claims(self, loss: float):
        """Method for recording unrecovered claims. If firm runs out of money it cannot _pay more claims and so that
            money is lost and recorded using this method. Called at start of dissolve to catch all instances necessary.
            Accepts:
                loss: Type integer, value of lost claim
            No return value"""
        self.cumulative_unrecovered_claims += loss

    def record_claims(self, claims: float):
        """This method records every claim made to insurers and reinsurers. It is called from both insurers and
            reinsurers (metainsuranceorg.py)."""
        # QUERY: Should insurance and reinsurance claims really be recorded together?
        self.cumulative_claims += claims

    def log(self):
        """Method to save the data of the simulation.
            No accepted values
            No return values
           Calls logger instance to save all the data of the simulation to a file, has to return if background run or
           not for replicating instances. This depends on parameters force_foreground and if the run is replicating
           or not."""
        self.logger.save_log(self.background_run)

    def compute_market_diffvar(self) -> float:
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
        # Real VaR is 1 for each firm, we think

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

    def get_unique_insurer_id(self) -> int:
        """Method for getting unique id for insurer. Used in initialising agents in start.py and insurancesimulation.
            Iterates after each call so id is unique to each firm.
           Returns:
                current_id: Type integer"""
        current_id = self.insurer_id_counter
        self.insurer_id_counter += 1
        return current_id

    def get_unique_reinsurer_id(self) -> int:
        """Method for getting unique id for insurer. Used in initialising agents in start.py and insurancesimulation.
            Iterates after each call so id is unique to each firm.
           Returns:
                current_id: Type integer"""
        current_id = self.reinsurer_id_counter
        self.reinsurer_id_counter += 1
        return current_id

    def get_unique_catbond_id(self) -> int:
        current_id = self.catbond_id_counter
        self.catbond_id_counter += 1
        return current_id

    def insurance_entry_index(self) -> int:
        """Method that returns the entry index for insurance firms, i.e. the index for the initial agent parameters
                   that is taken from the list of already created parameters.
               Returns:
                   Indices of the type of riskmodel that the least firms are using."""
        return self.insurance_models_counter[
            0 : self.simulation_parameters["no_riskmodels"]
        ].argmin()

    def reinsurance_entry_index(self) -> int:
        """Method that returns the entry index for reinsurance firms, i.e. the index for the initial agent parameters
                    that is taken from the list of already created parameters.
                Returns:
                    Indices of the type of riskmodel that the least reinsurance firms are using."""
        return self.reinsurance_models_counter[
            0 : self.simulation_parameters["no_riskmodels"]
        ].argmin()

    # noinspection PyMethodMayBeStatic
    def get_operational(self) -> bool:
        """Override get_operational to always return True, as the market will never die"""
        return True

    def reinsurance_capital_entry(self) -> float:
        # This method determines the capital market entry (initial cash) of reinsurers. It is only run in start.py.
        capital_per_non_re_cat = []

        for reinrisk in self.not_accepted_reinrisks:
            capital_per_non_re_cat.append(reinrisk.value)
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
        for firm in self.insurancefirms:
            firm.reset_pl()

        for reininsurancefirm in self.reinsurancefirms:
            reininsurancefirm.reset_pl()

        for cb in self.catbonds:
            cb.reset_pl()

    def get_risk_share(self, firm: "MetaInsuranceOrg") -> float:
        """Method to determine the total percentage of risks in the market that are held by a particular firm.
        For insurers uses insurance risks, for reinsurers uses reinsurance risks
        Calculates the
            Accepts:
                firm: an insurance or reinsurance firm
            Returns:
                proportion: type Float, the proportion of risks held by the given firm """
        if firm.is_insurer:
            total = self.simulation_parameters["no_risks"]
        elif firm.is_reinsurer:
            total = sum(
                [
                    reinfirm.number_underwritten_contracts()
                    for reinfirm in self.reinsurancefirms
                ]
                + [len(self.reinrisks)]
            )
        else:
            raise ValueError("Firm is neither insurer or reinsurer, which is odd")
        if total == 0:
            return 0
        else:
            return firm.number_underwritten_contracts() / total

    def get_total_firm_cash(self, firm_type: str):
        """Method to get sum of all cash of firms of a given type. Called from consider_buyout() but could be used for
        setting market premium.
            Accepts:
                type: Type String.
            Returns:
                sum_capital: Type Integer."""
        if firm_type == "insurer":
            sum_capital = sum([agent.get_cash() for agent in self.insurancefirms])
        elif firm_type == "reinsurer":
            sum_capital = sum([agent.get_cash() for agent in self.reinsurancefirms])
        else:
            raise ValueError(f"Recieved invalid firm type {firm_type}")
        return sum_capital

    def add_firm_to_be_sold(self, firm, time, reason):
        """Method to add firm to list of those being considered to buy dependant on firm type.
            Accepts:
                firm: Type Class.
                time: Type Integer.
                reason: Type String. Used in case of dissolution for logging.
            No return values."""
        if firm.is_insurer:
            self.selling_insurance_firms.append([firm, time, reason])
        elif firm.is_reinsurer:
            self.selling_reinsurance_firms.append([firm, time, reason])
        else:
            print("Not accepted type of firm")

    def get_firms_to_sell(self, firm_type):
        """Method to get list of firms that are up for selling based on type.
            Accepts:
               type: Type String.
            Returns:
               firms_info_sent: Type List of Lists. Contains firm, type and reason."""
        if firm_type == "insurer":
            firms_info_sent = [
                (firm, time, reason)
                for firm, time, reason in self.selling_insurance_firms
            ]
        elif firm_type == "reinsurer":
            firms_info_sent = [
                (firm, time, reason)
                for firm, time, reason in self.selling_reinsurance_firms
            ]
        else:
            raise ValueError(f"Unrecognised firm type {firm_type}")
        return firms_info_sent

    def remove_sold_firm(self, firm, time, reason):
        """Method to remove firm from list of firms being sold. Called when firm is bought buy another.
            Accepts:
                firm: Type Class.
                time: Type Integer.
                reason: Type String.
            No return values."""
        if firm.is_insurer:
            self.selling_insurance_firms.remove([firm, time, reason])
        elif firm.is_reinsurer:
            self.selling_reinsurance_firms.remove([firm, time, reason])

    def reset_selling_firms(self):
        """Method to reset list of firms being offered to sell. Called every iteration of insurance simulation.
            No accepted values.
            No return values.
        Firms being sold only considered for iteration they are added for given reason, after this not wanted so all
        are dissolved and relevant list attribute is reset."""
        for firm, time, reason in self.selling_insurance_firms:
            firm.dissolve(time, reason)
            for contract in firm.underwritten_contracts:
                contract.mature(time)
            firm.underwritten_contracts = []
        self.selling_insurance_firms = []

        for reinfirm, time, reason in self.selling_reinsurance_firms:
            reinfirm.dissolve(time, reason)
            for contract in reinfirm.underwritten_contracts:
                contract.mature(time)
            reinfirm.underwritten_contracts = []
        self.selling_reinsurance_firms = []

    def update_network_data(self):
        """Method to update the network data.
            No accepted values.
            No return values.
        This method is called from save_data() for every iteration to get the current adjacency list so network
        visualisation can be saved. Only called if conditions save_network is True and slim logs is False."""
        """obtain lists of operational entities"""
        op_entities = {}
        num_entities = {}
        for firmtype, firmlist in [
            ("insurers", self.insurancefirms),
            ("reinsurers", self.reinsurancefirms),
            ("catbonds", self.catbonds),
        ]:
            op_firmtype = [firm for firm in firmlist if firm.operational]
            op_entities[firmtype] = op_firmtype
            num_entities[firmtype] = len(op_firmtype)

        network_size = sum(num_entities.values())

        """Create weighted adjacency matrix and category edge labels"""
        weights_matrix = np.zeros(network_size ** 2).reshape(network_size, network_size)
        edge_labels = {}
        node_labels = {}
        for idx_to, firm in enumerate(
            op_entities["insurers"] + op_entities["reinsurers"]
        ):
            node_labels[idx_to] = firm.id
            eolrs = firm.get_excess_of_loss_reinsurance()
            for eolr in eolrs:
                try:
                    idx_from = num_entities["insurers"] + (
                        op_entities["reinsurers"] + op_entities["catbonds"]
                    ).index(eolr["reinsurer"])
                    weights_matrix[idx_from][idx_to] = eolr["value"]
                    edge_labels[idx_to, idx_from] = eolr["category"]
                except ValueError:
                    print("Reinsurer is not in list of reinsurance companies")

        """unweighted adjacency matrix"""
        adj_matrix = np.sign(weights_matrix)
        return adj_matrix.tolist(), node_labels, edge_labels, num_entities
