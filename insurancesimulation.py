from insurancefirm import InsuranceFirm
# from riskmodel import RiskModel
from reinsurancefirm import ReinsuranceFirm
from distributiontruncated import TruncatedDistWrapper
import numpy as np
import scipy.stats
import math
import sys, pdb
import isleconfig
import random
import copy
import logger

if isleconfig.show_network:
    import visualization_network


class InsuranceSimulation():
    def __init__(self, override_no_riskmodels, replic_ID, simulation_parameters, rc_event_schedule, rc_event_damage):
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
        self.number_riskmodels = simulation_parameters["no_riskmodels"]
        
        "Save parameters, sets parameters of sim according to isleconfig.py"
        if (replic_ID is None) or isleconfig.force_foreground:
            self.background_run = False 
        else:
            self.background_run = True
        self.replic_ID = replic_ID
        self.simulation_parameters = simulation_parameters
        
        "Unpacks parameters and sets distributions"
        self.catbonds_off = simulation_parameters["catbonds_off"]
        self.reinsurance_off = simulation_parameters["reinsurance_off"]
        self.total_no_risks = simulation_parameters["no_risks"]
        self.risk_factor_lower_bound = simulation_parameters["risk_factor_lower_bound"]
        self.cat_separation_distribution = scipy.stats.expon(0, simulation_parameters["event_time_mean_separation"])

        self.risk_factor_spread = simulation_parameters["risk_factor_upper_bound"] - simulation_parameters["risk_factor_lower_bound"]
        self.risk_factor_distribution = scipy.stats.uniform(loc=self.risk_factor_lower_bound, scale=self.risk_factor_spread)
        if not simulation_parameters["risk_factors_present"]:
            self.risk_factor_distribution = scipy.stats.uniform(loc=1.0, scale=0)
        self.risk_value_distribution = scipy.stats.uniform(loc=1000, scale=0)       #TODO is this correct?
        risk_factor_mean = self.risk_factor_distribution.mean()
        if np.isnan(risk_factor_mean):     # unfortunately scipy.stats.mean is not well-defined if scale = 0
            risk_factor_mean = self.risk_factor_distribution.rvs()
        non_truncated = scipy.stats.pareto(b=2, loc=0, scale=0.25)
        self.damage_distribution = TruncatedDistWrapper(lower_bound=0.25, upper_bound=1., dist=non_truncated)

        "set initial market price (normalized, i.e. must be multiplied by value or excess-deductible)"
        if self.simulation_parameters["expire_immediately"]:
            assert self.cat_separation_distribution.dist.name == "expon"
            expected_damage_frequency = 1 - scipy.stats.poisson(1 / self.simulation_parameters["event_time_mean_separation"] * \
                                                                self.simulation_parameters["mean_contract_runtime"]).pmf(0)
        else:
            expected_damage_frequency = self.simulation_parameters["mean_contract_runtime"] / \
                                                        self.cat_separation_distribution.mean()
        self.norm_premium = expected_damage_frequency * self.damage_distribution.mean() * \
                            risk_factor_mean * (1 + self.simulation_parameters["norm_profit_markup"])
        self.reinsurance_market_premium = self.market_premium = self.norm_premium

        "Set up monetary system (should instead be with the customers, if customers are modeled explicitly)"
        self.money_supply = self.simulation_parameters["money_supply"]
        self.obligations = []

        "Set up risk categories"
        self.riskcategories = list(range(self.simulation_parameters["no_categories"]))
        self.rc_event_schedule = []
        self.rc_event_damage = []
        self.rc_event_schedule_initial = []   # For debugging (cloud debugging) purposes is good to store the initial schedule of catastrophes
        self.rc_event_damage_initial = []     # and damages that will be use in a single run of the model.

        if rc_event_schedule is not None and rc_event_damage is not None:  # If we have schedules pass as arguments we used them.
            self.rc_event_schedule = copy.copy(rc_event_schedule)
            self.rc_event_schedule_initial = copy.copy(rc_event_schedule)

            self.rc_event_damage = copy.copy(rc_event_damage)
            self.rc_event_damage_initial = copy.copy(rc_event_damage)
        else:                                                      # Otherwise the schedules and damages are generated.
            self.setup_risk_categories_caller()

        "Set up risks"
        risk_value_mean = self.risk_value_distribution.mean()
        if np.isnan(risk_value_mean):                 # unfortunately scipy.stats.mean is not well-defined if scale = 0
            risk_value_mean = self.risk_value_distribution.rvs()
        rrisk_factors = self.risk_factor_distribution.rvs(size=self.simulation_parameters["no_risks"])
        rvalues = self.risk_value_distribution.rvs(size=self.simulation_parameters["no_risks"])
        rcategories = np.random.randint(0, self.simulation_parameters["no_categories"], size=self.simulation_parameters["no_risks"])
        self.risks = [{"risk_factor": rrisk_factors[i], "value": rvalues[i], "category": rcategories[i], "owner": self} for i in range(self.simulation_parameters["no_risks"])]
        self.risks_counter = [0,0,0,0]

        for item in self.risks:
            self.risks_counter[item["category"]] = self.risks_counter[item["category"]] + 1

        self.inaccuracy = self.get_all_riskmodel_combinations(self.simulation_parameters["riskmodel_inaccuracy_parameter"])

        self.inaccuracy = random.sample(self.inaccuracy, self.simulation_parameters["no_riskmodels"])

        risk_model_configurations = [{"damage_distribution": self.damage_distribution,
                                      "expire_immediately": self.simulation_parameters["expire_immediately"],
                                      "cat_separation_distribution": self.cat_separation_distribution,
                                      "norm_premium": self.norm_premium,
                                      "no_categories": self.simulation_parameters["no_categories"],
                                      "risk_value_mean": risk_value_mean,
                                      "risk_factor_mean": risk_factor_mean,
                                      "norm_profit_markup": self.simulation_parameters["norm_profit_markup"],
                                      "margin_of_safety": self.simulation_parameters["riskmodel_margin_of_safety"],
                                      "var_tail_prob": self.simulation_parameters["value_at_risk_tail_probability"],
                                      "inaccuracy_by_categ": self.inaccuracy[i]} \
                                      for i in range(self.simulation_parameters["no_riskmodels"])]
        
        "Setting up agents (to be done from start.py)"
        self.agent_parameters = {"insurancefirm": [], "reinsurancefirm": []}
        self.initialize_agent_parameters("insurancefirm", simulation_parameters, risk_model_configurations)
        self.initialize_agent_parameters("reinsurancefirm", simulation_parameters, risk_model_configurations)

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
        self.logger = logger.Logger(no_riskmodels=simulation_parameters["no_riskmodels"], 
                                    rc_event_schedule_initial=self.rc_event_schedule_initial, 
                                    rc_event_damage_initial=self.rc_event_damage_initial)
        
        self.insurance_models_counter = np.zeros(self.simulation_parameters["no_categories"])
        self.reinsurance_models_counter = np.zeros(self.simulation_parameters["no_categories"])

    def initialize_agent_parameters(self, firmtype, simulation_parameters, risk_model_configurations):
        """General function for initialising the agent parameters
            Takes the firm type as argument, also needing sim params and risk configs
              Creates the agent parameters of both firm types for the initial number specified in isleconfig.py
                Returns None"""
        if firmtype == "insurancefirm":
            self.insurer_id_counter = 0
            no_firms = simulation_parameters["no_insurancefirms"]
            initial_cash = "initial_agent_cash"
            reinsurance_level_lowerbound = simulation_parameters["insurance_reinsurance_levels_lower_bound"]
            reinsurance_level_upperbound = simulation_parameters["insurance_reinsurance_levels_upper_bound"]

        elif firmtype == "reinsurancefirm":
            self.reinsurer_id_counter = 0
            no_firms = simulation_parameters["no_reinsurancefirms"]
            initial_cash = "initial_reinagent_cash"
            reinsurance_level_lowerbound = simulation_parameters["reinsurance_reinsurance_levels_lower_bound"]
            reinsurance_level_upperbound = simulation_parameters["reinsurance_reinsurance_levels_upper_bound"]

        for i in range(no_firms):
            if simulation_parameters['static_non-proportional_reinsurance_levels']:
                reinsurance_level = simulation_parameters["default_non-proportional_reinsurance_deductible"]
            else:
                reinsurance_level = np.random.uniform(reinsurance_level_lowerbound, reinsurance_level_upperbound)

            riskmodel_config = risk_model_configurations[i % len(risk_model_configurations)]
            self.agent_parameters[firmtype].append({'id': self.get_unique_insurer_id(), 'initial_cash': simulation_parameters[initial_cash],
                'riskmodel_config': riskmodel_config, 'norm_premium': self.norm_premium,
                'profit_target': simulation_parameters["norm_profit_markup"],
                'initial_acceptance_threshold': simulation_parameters["initial_acceptance_threshold"],
                'acceptance_threshold_friction': simulation_parameters["acceptance_threshold_friction"],
                'reinsurance_limit': simulation_parameters["reinsurance_limit"],
                'non-proportional_reinsurance_level': reinsurance_level,
                'capacity_target_decrement_threshold': simulation_parameters['capacity_target_decrement_threshold'],
                'capacity_target_increment_threshold': simulation_parameters['capacity_target_increment_threshold'],
                'capacity_target_decrement_factor': simulation_parameters['capacity_target_decrement_factor'],
                'capacity_target_increment_factor': simulation_parameters['capacity_target_increment_factor'],
                'interest_rate': simulation_parameters["interest_rate"]})

    def build_agents(self, agent_class, agent_class_string, parameters, agent_parameters):
        """Method for building new agents, only used for re/insurance firms. Loops through the agent parameters for each
            initialised agent to create an instance of them using re/insurancefirm.
            Accepts:
                Agent_class: class of agent, either InsuranceFirm or ReinsuranceFirm.
                agent_class_string: String Type containing string of agent class. Not used.
                parameters: DataDict, contains config parameters.
                agent_parameters: DataDict of agent parameters.
            Returns:
                agents: List Type, list of agent class instances created by loop"""
        agents = []
        for ap in agent_parameters:
            agents.append(agent_class(parameters, ap))
        return agents
        
    def accept_agents(self, agent_class_string, agents, agent_group=None, time=0):
        """Method to 'accept' agents in that it adds agent to relevant list of agents kept by simulation
            instance, also adds agent to logger. Also takes created agents initial cash out of economy.
            Accepts:
                agent_class_string: String Type.
                agents: List type of agent class instances.
                agent_group: List type of agent class instances.
                time: Integer type, not used
            Returns:
                None"""
        # TODO: fix agent id's for late entrants (both firms and catbonds)
        if agent_class_string == "insurancefirm":
            try:
                self.insurancefirms += agents
                self.insurancefirms_group = agent_group
            except:
                print(sys.exc_info())
                pdb.set_trace()
            # fix self.history_logs['individual_contracts'] list
            for agent in agents:
                self.logger.add_insurance_agent()
            # remove new agent cash from simulation cash to ensure stock flow consistency
            new_agent_cash = sum([agent.cash for agent in agents])
            self.reduce_money_supply(new_agent_cash)
        elif agent_class_string == "reinsurancefirm":
            try:
                self.reinsurancefirms += agents
                self.reinsurancefirms_group = agent_group
            except:
                print(sys.exc_info())
                pdb.set_trace()
            # remove new agent cash from simulation cash to ensure stock flow consistency
            new_agent_cash = sum([agent.cash for agent in agents])
            self.reduce_money_supply(new_agent_cash)
        elif agent_class_string == "catbond":
            try:
                self.catbonds += agents
            except:
                print(sys.exc_info())
                pdb.set_trace()            
        else:
            assert False, "Error: Unexpected agent class used {0:s}".format(agent_class_string)

    def delete_agents(self, agent_class_string, agents):
        """Method for deleting catbonds as it is only agent that is allowed to be removed
            alters lists of catbonds
              Returns none"""
        if agent_class_string == "catbond":
            for agent in agents:
                self.catbonds.remove(agent)
        else:
            assert False, "Trying to remove unremovable agent, type: {0:s}".format(agent_class_string)
    
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
            print("\rTime: {0:4d}".format(t), end="")

        self.reset_pls()

        # Adjust market premiums
        sum_capital = sum([agent.get_cash() for agent in self.insurancefirms])
        self.adjust_market_premium(capital=sum_capital)
        sum_capital = sum([agent.get_cash() for agent in self.reinsurancefirms])
        self.adjust_reinsurance_market_premium(capital=sum_capital)

        # Pay obligations
        self.effect_payments(t)
        
        # Identify perils and effect claims
        for categ_id in range(len(self.rc_event_schedule)):
            try:
                if len(self.rc_event_schedule[categ_id]) > 0:
                    assert self.rc_event_schedule[categ_id][0] >= t
            except:
                print("Something wrong; past events not deleted", file=sys.stderr)
            if len(self.rc_event_schedule[categ_id]) > 0 and self.rc_event_schedule[categ_id][0] == t:
                self.rc_event_schedule[categ_id] = self.rc_event_schedule[categ_id][1:]
                damage_extent = copy.copy(self.rc_event_damage[categ_id][0])       # Schedules of catastrophes and damages must be generated at the same time.
                self.inflict_peril(categ_id=categ_id, damage=damage_extent, t=t) # TODO: consider splitting the following lines from this method and running it with nb.jit
                self.rc_event_damage[categ_id] = self.rc_event_damage[categ_id][1:]
            else:
                if isleconfig.verbose:
                    print("Next peril ", self.rc_event_schedule[categ_id])
        
        # Shuffle risks (insurance and reinsurance risks)
        self.shuffle_risks()

        # Reset reinweights
        self.reset_reinsurance_weights()
                    
        # Iterate reinsurnace firm agents
        for reinagent in self.reinsurancefirms:
            reinagent.iterate(t)

        self.reinrisks = []

        # Reset weights
        self.reset_insurance_weights()
                    
        # Iterate insurance firm agents
        for agent in self.insurancefirms:
            agent.iterate(t)
        
        # Iterate catbonds
        for agent in self.catbonds:
            agent.iterate(t)

        self.insurance_models_counter = np.zeros(self.simulation_parameters["no_categories"])

        for insurer in self.insurancefirms:
            for i in range(len(self.inaccuracy)):
                if insurer.operational:
                    if insurer.riskmodel.inaccuracy == self.inaccuracy[i]:
                        self.insurance_models_counter[i] += 1

        self.reinsurance_models_counter = np.zeros(self.simulation_parameters["no_categories"])

        for reinsurer in self.reinsurancefirms:
            for i in range(len(self.inaccuracy)):
                if reinsurer.operational:
                    if reinsurer.riskmodel.inaccuracy == self.inaccuracy[i]:
                        self.reinsurance_models_counter[i] += 1

        if isleconfig.show_network and t % 40 == 0 and t > 0:
            RN = visualization_network.ReinsuranceNetwork(self.insurancefirms, self.reinsurancefirms, self.catbonds)
            RN.compute_measures()
            RN.visualize()

    def save_data(self):
        """Method to collect statistics about the current state of the simulation. Will pass these to the 
           Logger object (self.logger) to be recorded.
            No arguments.
            Returns None."""
        
        """ collect data """
        total_cash_no = sum([insurancefirm.cash for insurancefirm in self.insurancefirms])
        total_excess_capital = sum([insurancefirm.get_excess_capital() for insurancefirm in self.insurancefirms])
        total_profitslosses =  sum([insurancefirm.get_profitslosses() for insurancefirm in self.insurancefirms])
        total_contracts_no = sum([len(insurancefirm.underwritten_contracts) for insurancefirm in self.insurancefirms])
        total_reincash_no = sum([reinsurancefirm.cash for reinsurancefirm in self.reinsurancefirms])
        total_reinexcess_capital = sum([reinsurancefirm.get_excess_capital() for reinsurancefirm in self.reinsurancefirms])
        total_reinprofitslosses =  sum([reinsurancefirm.get_profitslosses() for reinsurancefirm in self.reinsurancefirms])
        total_reincontracts_no = sum([len(reinsurancefirm.underwritten_contracts) for reinsurancefirm in self.reinsurancefirms])
        operational_no = sum([insurancefirm.operational for insurancefirm in self.insurancefirms])
        reinoperational_no = sum([reinsurancefirm.operational for reinsurancefirm in self.reinsurancefirms])
        catbondsoperational_no = sum([catbond.operational for catbond in self.catbonds])
        
        """ collect agent-level data """
        insurance_firms = [(insurancefirm.cash,insurancefirm.id,insurancefirm.operational) for insurancefirm in self.insurancefirms]
        reinsurance_firms = [(reinsurancefirm.cash,reinsurancefirm.id,reinsurancefirm.operational) for reinsurancefirm in self.reinsurancefirms]
        
        """ prepare dict """
        current_log = {}
        current_log['total_cash'] = total_cash_no
        current_log['total_excess_capital'] = total_excess_capital
        current_log['total_profitslosses'] = total_profitslosses
        current_log['total_contracts'] = total_contracts_no
        current_log['total_operational'] = operational_no
        current_log['total_reincash'] = total_reincash_no
        current_log['total_reinexcess_capital'] = total_reinexcess_capital
        current_log['total_reinprofitslosses'] = total_reinprofitslosses
        current_log['total_reincontracts'] = total_reincontracts_no
        current_log['total_reinoperational'] = reinoperational_no
        current_log['total_catbondsoperational'] = catbondsoperational_no
        current_log['market_premium'] = self.market_premium
        current_log['market_reinpremium'] = self.reinsurance_market_premium
        current_log['cumulative_bankruptcies'] = self.cumulative_bankruptcies
        current_log['cumulative_market_exits'] = self.cumulative_market_exits
        current_log['cumulative_unrecovered_claims'] = self.cumulative_unrecovered_claims
        current_log['cumulative_claims'] = self.cumulative_claims    #Log the cumulative claims received so far.
        
        """ add agent-level data to dict""" 
        current_log['insurance_firms_cash'] = insurance_firms
        current_log['reinsurance_firms_cash'] = reinsurance_firms
        current_log['market_diffvar'] = self.compute_market_diffvar()
        
        current_log['individual_contracts'] = []
        individual_contracts_no = [len(insurancefirm.underwritten_contracts) for insurancefirm in self.insurancefirms]
        for i in range(len(individual_contracts_no)):
            current_log['individual_contracts'].append(individual_contracts_no[i])

        """ call to Logger object """
        self.logger.record_data(current_log)
        
    def obtain_log(self, requested_logs=None):
        """This function allows to return in a list all the data generated by the model. There is no other way to
            transfer it back from the cloud."""
        return self.logger.obtain_log(requested_logs)
    
    def finalize(self, *args):
        """Function to handle operations after the end of the simulation run.
           Currently empty.
           It may be used to handle e.g. logging by including:
            self.log()
           but logging has been moved to start.py and ensemble.py
           """
        pass

    def inflict_peril(self, categ_id, damage, t):
        """Method that calculates percentage damage done to each underwritten risk that is affected in the category
            that event happened in. Passes values to allow calculation contracts to be resolved.
            Arguments:
                ID of category events took place
                Given severity of damage from pareto distribution
                Time iteration
            No return value"""
        affected_contracts = [contract for insurer in self.insurancefirms for contract in insurer.underwritten_contracts if contract.category == categ_id]
        if isleconfig.verbose:
            print("**** PERIL ", damage)
        damagevalues = np.random.beta(1, 1./damage -1, size=self.risks_counter[categ_id])
        uniformvalues = np.random.uniform(0, 1, size=self.risks_counter[categ_id])
        [contract.explode(t, uniformvalues[i], damagevalues[i]) for i, contract in enumerate(affected_contracts)]
    
    def receive_obligation(self, amount, recipient, due_time, purpose):
        """Method for adding obligation to list that is resolved at the start if each iteration of simulation. Only
            called by metainsuranceorg for adding interest to cash.
            Arguments
                Amount: obligation value
                Recipient: Who obligation is owed to
                Due Time
                Purpose: Reason for obligation (Interest due)
            Returns None"""

        obligation = {"amount": amount, "recipient": recipient, "due_time": due_time, "purpose": purpose}
        self.obligations.append(obligation)

    def effect_payments(self, time):
        """Method for checking and paying obligation if due.
            Arguments
                Current time to allow check if due
            Returns None"""
        due = [item for item in self.obligations if item["due_time"]<=time]
        self.obligations = [item for item in self.obligations if item["due_time"]>time]
        sum_due = sum([item["amount"] for item in due])
        for obligation in due:
            self.pay(obligation)

    def pay(self, obligation):
        """Method for paying obligations called from effect_payments
            Accepts:
                Obligation: Type DataDict with categories amount, recipient, due time, purpose.
            Returns None"""
        amount = obligation["amount"]
        recipient = obligation["recipient"]
        purpose = obligation["purpose"]
        try:
            assert self.money_supply > amount
        except:
            print("Something wrong: economy out of money", file=sys.stderr)
        if self.get_operational() and recipient.get_operational():
            self.money_supply -= amount
            recipient.receive(amount)

    def receive(self, amount):
        """Method to accept cash payments. As insurance simulation cash is economy, adds money to total economy.
            Accepts:
                Amount due: Type Integer
            Returns None"""
        self.money_supply += amount

    def reduce_money_supply(self, amount):
        """Method to reduce money supply immediately and without payment recipient
            (used to adjust money supply to compensate for agent endowment).
            Accepts:
                amount: Type Integer"""
        self.money_supply -= amount
        assert self.money_supply >= 0

    def reset_reinsurance_weights(self):
        """Method for clearing and setting reinsurance weights dependant on how many reinsurance companies exist and
            how many offered reinsurance risks there are."""
        self.not_accepted_reinrisks = []

        operational_reinfirms = [reinsurancefirm for reinsurancefirm in self.reinsurancefirms if reinsurancefirm.operational]

        operational_no = len(operational_reinfirms)

        reinrisks_no = len(self.reinrisks)

        self.reinsurers_weights = {}

        for reinsurer in self.reinsurancefirms:
            self.reinsurers_weights[reinsurer.id] = 0

        if operational_no > 0:

            if reinrisks_no/operational_no > 1:
                weights = reinrisks_no/operational_no
                for reinsurer in self.reinsurancefirms:
                    self.reinsurers_weights[reinsurer.id] = math.floor(weights)
            else:
                for i in range(len(self.reinrisks)):
                    s = math.floor(np.random.uniform(0, len(operational_reinfirms), 1))
                    self.reinsurers_weights[operational_reinfirms[s].id] += 1
        else:
            self.not_accepted_reinrisks = self.reinrisks

    def reset_insurance_weights(self):
        """Method for clearing and setting insurance weights dependant on how many insurance companies exist and
            how many insurance risks are offered. This determined which risks are sent to metainsuranceorg
            iteration."""
        operational_no = sum([insurancefirm.operational for insurancefirm in self.insurancefirms])

        operational_firms = [insurancefirm for insurancefirm in self.insurancefirms if insurancefirm.operational]

        risks_no = len(self.risks)

        self.insurers_weights = {}

        for insurer in self.insurancefirms:
            self.insurers_weights[insurer.id] = 0

        if operational_no > 0:

            if risks_no/operational_no > 1:
                weights = risks_no/operational_no
                for insurer in self.insurancefirms:
                    self.insurers_weights[insurer.id] = math.floor(weights)
            else:
                for i in range(len(self.risks)):
                    s = math.floor(np.random.uniform(0, len(operational_firms), 1))
                    self.insurers_weights[operational_firms[s].id] += 1

    def shuffle_risks(self):
        """Method for shuffling risks."""
        np.random.shuffle(self.reinrisks)
        np.random.shuffle(self.risks)

    def adjust_market_premium(self, capital):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the insurance market (insurance only).
               No return value.
           This method adjusts the premium charged by insurance firms for the risks covered. The premium reduces linearly
           with the capital available in the insurance market and viceversa. The premium reduces until it reaches a minimum
           below which no insurer is willing to reduce further the price. This method is only called in the self.iterate()
           method of this class."""
        self.market_premium = self.norm_premium * (self.simulation_parameters["upper_price_limit"] - self.simulation_parameters["premium_sensitivity"] * capital / (self.simulation_parameters["initial_agent_cash"] * self.damage_distribution.mean() * self.simulation_parameters["no_risks"]))
        if self.market_premium < self.norm_premium * self.simulation_parameters["lower_price_limit"]:
            self.market_premium = self.norm_premium * self.simulation_parameters["lower_price_limit"]
    
    def adjust_reinsurance_market_premium(self, capital):
        """Adjust_market_premium Method.
               Accepts arguments
                   capital: Type float. The total capital (cash) available in the reinsurance market (reinsurance only).
               No return value.
           This method adjusts the premium charged by reinsurance firms for the risks covered. The premium reduces linearly
           with the capital available in the reinsurance market and viceversa. The premium reduces until it reaches a minimum
           below which no reinsurer is willing to reduce further the price. This method is only called in the self.iterate()
           method of this class."""
        self.reinsurance_market_premium = self.norm_premium * (self.simulation_parameters["upper_price_limit"] - self.simulation_parameters["reinpremium_sensitivity"] * capital / (self.simulation_parameters["initial_agent_cash"] * self.damage_distribution.mean() * self.simulation_parameters["no_risks"]))
        if self.reinsurance_market_premium < self.norm_premium * self.simulation_parameters["lower_price_limit"]:
            self.reinsurance_market_premium = self.norm_premium * self.simulation_parameters["lower_price_limit"]

    def get_market_premium(self):
        """Get_market_premium Method.
               Accepts no arguments.
               Returns:
                   self.market_premium: Type float. The current insurance market premium.
           This method returns the current insurance market premium."""
        return self.market_premium

    def get_market_reinpremium(self):
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
        # TODO: make premiums dependend on the deductible per value (np_reinsurance_deductible_fraction) -> DONE.
        # TODO: make max_reduction into simulation_parameter ?
        if self.reinsurance_off:
            return float('inf')
        max_reduction = 0.1
        return self.reinsurance_market_premium * (1. - max_reduction * np_reinsurance_deductible_fraction)
        
    def get_cat_bond_price(self, np_reinsurance_deductible_fraction):
        # TODO: implement function dependent on total capital in cat bonds and on deductible ()
        # TODO: make max_reduction and max_CB_surcharge into simulation_parameters ?
        if self.catbonds_off:
            return float('inf')
        max_reduction = 0.9
        max_CB_surcharge = 0.5 
        return self.reinsurance_market_premium * (1. + max_CB_surcharge - max_reduction * np_reinsurance_deductible_fraction)
        
    def append_reinrisks(self, item):
        """Method for appending reinrisks to simulation instance. Called from insurancefirm
            Accepts: item (Type: List)"""
        if len(item) > 0:
            self.reinrisks.append(item)

    def get_reinrisks(self):
        """Method for shuffling reinsurance risks
            Returns: reinsurance risks"""
        np.random.shuffle(self.reinrisks)
        return self.reinrisks

    def solicit_insurance_requests(self, id, cash, insurer):
        """Method for determining which risks are to be assessed by firms based on insurer weights
            Accepts:
                id: Type integer
                cash: Type Integer
                insurer: Type firm metainsuranceorg instance
            Returns:
                risks_to_be_sent: Type List"""
        risks_to_be_sent = self.risks[:int(self.insurers_weights[insurer.id])]
        self.risks = self.risks[int(self.insurers_weights[insurer.id]):]
        for risk in insurer.risks_kept:
            risks_to_be_sent.append(risk)

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
        reinrisks_to_be_sent = self.reinrisks[:int(self.reinsurers_weights[reinsurer.id])]
        self.reinrisks = self.reinrisks[int(self.reinsurers_weights[reinsurer.id]):]

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
        # TODO:Remove?
        self.not_accepted_reinrisks += not_accepted_risks

    def get_all_riskmodel_combinations(self, rm_factor):
        """Method  for calculating riskmodels for each category based on the risk model inaccuracy parameter, and is
            used purely to assign inaccuracy. Currently all equal and overwritten immediately.
            Accepts:
                rm_factor: Type Integer = risk model inaccuracy parameter
            Returns:
                riskmodels: Type list"""
        riskmodels = []
        for i in range(self.simulation_parameters["no_categories"]):
            riskmodel_combination = rm_factor * np.ones(self.simulation_parameters["no_categories"])
            riskmodel_combination[i] = 1/rm_factor
            riskmodels.append(riskmodel_combination.tolist())
        return riskmodels

    def setup_risk_categories(self):
        """Method for generating the schedule of events and the percentage damage/severity caused by the event.
            Only called if risk categories have not already been set as want to keep equal to allow for comparison.
                Both must also be calculated at the same time to allow for replication.
                    Event schedule and damage based on distributions set in __init__."""
        for i in self.riskcategories:
            event_schedule = []
            event_damage = []
            total = 0
            while total < self.simulation_parameters["max_time"]:
                separation_time = self.cat_separation_distribution.rvs()
                total += int(math.ceil(separation_time))
                if total < self.simulation_parameters["max_time"]:
                    event_schedule.append(total)
                    event_damage.append(self.damage_distribution.rvs())   #Schedules of catastrophes and damages must me generated at the same time. Reason: replication across different risk models.
            self.rc_event_schedule.append(event_schedule)
            self.rc_event_damage.append(event_damage)

        self.rc_event_schedule_initial = copy.copy(self.rc_event_damage)   #For debugging (cloud debugging) purposes is good to store the initial schedule of catastrophes
        self.rc_event_damage_initial = copy.copy(self.rc_event_damage)     #and damages that will be use in a single run of the model.

    def setup_risk_categories_caller(self):
        """Method for calling setup_risk_categories. If conditions are set such that the system is replicating it is
            not called otherwise calls setup."""
        if self.replic_ID is not None:
            if isleconfig.replicating:
                self.restore_state_and_risk_categories()
            else:
                self.setup_risk_categories()
                self.save_state_and_risk_categories()
        else:
            self.setup_risk_categories()

    def save_state_and_risk_categories(self):
        """Method to save numpy Mersenne Twister state and event schedule to allow for replication and continuation."""
        mersennetwoster_randomseed = str(np.random.get_state())
        mersennetwoster_randomseed = mersennetwoster_randomseed.replace("\n","").replace("array", "np.array").replace("uint32", "np.uint32")
        wfile = open("data/replication_randomseed.dat","a")
        wfile.write(mersennetwoster_randomseed+"\n")
        wfile.close()

        wfile = open("data/replication_rc_event_schedule.dat","a")
        wfile.write(str(self.rc_event_schedule)+"\n")
        wfile.close()
        
    def restore_state_and_risk_categories(self):
        """Method to access saved event schedule, seed, and Mersenne twister state to allow for continuation."""
        rfile = open("data/replication_rc_event_schedule.dat","r")
        found = False
        for i, line in enumerate(rfile):
            if i == self.replic_ID:
                self.rc_event_schedule = eval(line)
                found = True
        rfile.close()
        assert found, "rc event schedule for current replication ID number {0:d} not found in data file. Exiting.".format(self.replic_ID)
        rfile = open("data/replication_randomseed.dat","r")
        found = False
        for i, line in enumerate(rfile):
            #print(i, self.replic_ID)
            if i == self.replic_ID:
                mersennetwister_randomseed = eval(line)
                found = True
        rfile.close()
        np.random.set_state(mersennetwister_randomseed)
        assert found, "mersennetwister randomseed for current replication ID number {0:d} not found in data file. Exiting.".format(self.replic_ID)

    def insurance_firm_market_entry(self, agent_type="InsuranceFirm"):
        """Method to determine if re/insurance firm enters the market based on set entry probabilities and a random
            integer generated between 0, 1.
            Accepts:
                agent_type: Type String
            Returns:
                 True if firm can enter market
                 False if firm cannot enter market"""

        if agent_type == "InsuranceFirm":
            prob = self.simulation_parameters["insurance_firm_market_entry_probability"]
        elif agent_type == "ReinsuranceFirm":
            prob = self.simulation_parameters["reinsurance_firm_market_entry_probability"]
        else:
            assert False, "Unknown agent type. Simulation requested to create agent of type {0:s}".format(agent_type)
        if np.random.random() < prob:
            return True
        else:
            return False

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
        """Method to log if a background run or not dependant on parameters force_foreground and if the run is
            replicating or not."""
        self.logger.save_log(self.background_run)
        
    def compute_market_diffvar(self):
        """Method for calculating difference between number of all firms and the total value at risk. Used only in save
            data when adding to the logger data dict."""
        varsfirms = []
        for firm in self.insurancefirms:
            if firm.operational:
                varsfirms.append(firm.var_counter_per_risk)
        totalina = sum(varsfirms)

        varsfirms = []
        for firm in self.insurancefirms:
            if firm.operational:
                varsfirms.append(1)
        totalreal = sum(varsfirms)

        varsreinfirms = []
        for reinfirm in self.reinsurancefirms:
            if reinfirm.operational:
                varsreinfirms.append(reinfirm.var_counter_per_risk)
        totalina = totalina + sum(varsreinfirms)

        varsreinfirms = []
        for reinfirm in self.reinsurancefirms:
            if reinfirm.operational:
                varsreinfirms.append(1)
        totalreal = totalreal + sum(varsreinfirms)

        totaldiff = totalina - totalreal
        
        return totaldiff
        #self.history_logs['market_diffvar'].append(totaldiff)

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
        return self.insurance_models_counter[0:self.simulation_parameters["no_riskmodels"]].argmin()

    def reinsurance_entry_index(self):
        """Method that returns the entry index for reinsurance firms, i.e. the index for the initial agent parameters
            that is taken from the list of already created parameters.
        Returns:
            Indices of the type of riskmodel that the least reinsurance firms are using."""
        return self.reinsurance_models_counter[0:self.simulation_parameters["no_riskmodels"]].argmin()

    def get_operational(self):
        """Method to return if simulation is operational. Always true. Used only in pay methods above and
            metainsuranceorg.
           Accepts no arguments
           Returns True"""
        return True

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

