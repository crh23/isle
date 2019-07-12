import isleconfig
import numpy as np
import scipy.stats
import copy
import math
from insurancecontract import InsuranceContract
from reinsurancecontract import ReinsuranceContract
from riskmodel import RiskModel
import functools


def get_mean(x):
    return sum(x) / len(x)


# A quick check tells me that we don't need a very large cache for this, as it only tends to repeat a couple of times.
@functools.lru_cache(maxsize=16)
def get_mean_std(x):
    # At the moment this is always called with a no_category length array
    # I have tested the numpy versions of this, they are slower for small arrays but much, much faster for large ones
    # If we ever let no_category be much larger, might want to use np for this bit
    m = get_mean(x)
    std = math.sqrt(sum((val - m) ** 2 for val in x)) / len(x)
    return m, std


class MetaInsuranceOrg:
    def __init__(self, simulation_parameters, agent_parameters):
        """Constructor method.
                    Accepts:
                        Simulation_parameters: Type DataDict
                        agent_parameters:   Type DataDict
                    Constructor creates general instance of an insurance company which is inherited by the reinsurance and
                    insurance firm classes. Initialises all necessary values provided by config file."""
        self.simulation = simulation_parameters["simulation"]
        self.simulation_parameters = simulation_parameters
        self.contract_runtime_dist = scipy.stats.randint(
            simulation_parameters["mean_contract_runtime"]
            - simulation_parameters["contract_runtime_halfspread"],
            simulation_parameters["mean_contract_runtime"]
            + simulation_parameters["contract_runtime_halfspread"]
            + 1,
        )
        self.default_contract_payment_period = simulation_parameters[
            "default_contract_payment_period"
        ]
        self.id = agent_parameters["id"]
        self.cash = agent_parameters["initial_cash"]
        self.capacity_target = self.cash * 0.9
        self.capacity_target_decrement_threshold = agent_parameters[
            "capacity_target_decrement_threshold"
        ]
        self.capacity_target_increment_threshold = agent_parameters[
            "capacity_target_increment_threshold"
        ]
        self.capacity_target_decrement_factor = agent_parameters[
            "capacity_target_decrement_factor"
        ]
        self.capacity_target_increment_factor = agent_parameters[
            "capacity_target_increment_factor"
        ]
        self.excess_capital = self.cash
        self.premium = agent_parameters["norm_premium"]
        self.profit_target = agent_parameters["profit_target"]
        self.acceptance_threshold = agent_parameters[
            "initial_acceptance_threshold"
        ]  # 0.5
        self.acceptance_threshold_friction = agent_parameters[
            "acceptance_threshold_friction"
        ]  # 0.9 #1.0 to switch off
        self.interest_rate = agent_parameters["interest_rate"]
        self.reinsurance_limit = agent_parameters["reinsurance_limit"]
        self.simulation_no_risk_categories = simulation_parameters["no_categories"]
        self.simulation_reinsurance_type = simulation_parameters[
            "simulation_reinsurance_type"
        ]
        self.dividend_share_of_profits = simulation_parameters[
            "dividend_share_of_profits"
        ]

        self.owner = self.simulation  # TODO: Make this into agent_parameter value?
        self.per_period_dividend = 0
        self.cash_last_periods = list(np.zeros(4, dtype=int) * self.cash)

        rm_config = agent_parameters["riskmodel_config"]

        """Here we modify the margin of safety depending on the number of risks models available in the market. 
           When is 0 all risk models have the same margin of safety. The reason for doing this is that with more risk
           models the firms tend to be closer to the max capacity"""
        margin_of_safety_correction = (
            rm_config["margin_of_safety"]
            + (simulation_parameters["no_riskmodels"] - 1)
            * simulation_parameters["margin_increase"]
        )

        self.riskmodel = RiskModel(
            damage_distribution=rm_config["damage_distribution"],
            expire_immediately=rm_config["expire_immediately"],
            cat_separation_distribution=rm_config["cat_separation_distribution"],
            norm_premium=rm_config["norm_premium"],
            category_number=rm_config["no_categories"],
            init_average_exposure=rm_config["risk_value_mean"],
            init_average_risk_factor=rm_config["risk_factor_mean"],
            init_profit_estimate=rm_config["norm_profit_markup"],
            margin_of_safety=margin_of_safety_correction,
            var_tail_prob=rm_config["var_tail_prob"],
            inaccuracy=rm_config["inaccuracy_by_categ"],
        )

        self.category_reinsurance = [
            None for i in range(self.simulation_no_risk_categories)
        ]
        if self.simulation_reinsurance_type == "non-proportional":
            if agent_parameters["non-proportional_reinsurance_level"] is not None:
                self.np_reinsurance_deductible_fraction = agent_parameters[
                    "non-proportional_reinsurance_level"
                ]
            else:
                self.np_reinsurance_deductible_fraction = simulation_parameters[
                    "default_non-proportional_reinsurance_deductible"
                ]
            self.np_reinsurance_excess_fraction = simulation_parameters[
                "default_non-proportional_reinsurance_excess"
            ]
            self.np_reinsurance_premium_share = simulation_parameters[
                "default_non-proportional_reinsurance_premium_share"
            ]
        self.obligations = []
        self.underwritten_contracts = []
        self.profits_losses = 0
        # self.reinsurance_contracts = []
        self.operational = True
        self.is_insurer = True
        self.is_reinsurer = False

        """set up risk value estimate variables"""
        self.var_counter = 0  # sum over risk model inaccuracies for all contracts
        self.var_counter_per_risk = 0  # average risk model inaccuracy across contracts
        self.var_sum = 0  # sum over initial VaR for all contracts
        self.counter_category = np.zeros(
            self.simulation_no_risk_categories
        )  # var_counter disaggregated by category
        self.var_category = np.zeros(
            self.simulation_no_risk_categories
        )  # var_sum disaggregated by category
        self.naccep = []
        self.risks_kept = []
        self.reinrisks_kept = []
        self.balance_ratio = simulation_parameters["insurers_balance_ratio"]
        self.recursion_limit = simulation_parameters["insurers_recursion_limit"]
        # QUERY: Should this have to sum to self.cash
        self.cash_left_by_categ =
            self.cash * np.ones(
            self.simulation_parameters["no_categories"]
        )
        self.market_permanency_counter = 0

    def iterate(self, time):
        """Method that iterates each firm by one time step.
                    Accepts:
                        Time: Type Integer
                    No return value
                    For each time step this method obtains every firms interest payments, pays obligations, claim reinsurance,
                    matures necessary contracts. Check condition for operational firms (as never removed) so only operational
                    firms receive new risks to evaluate, pay dividends, adjust capacity."""

        """obtain investments yield"""
        self.obtain_yield(time)

        """realize due payments"""
        self.effect_payments(time)
        if isleconfig.verbose:
            print(
                time,
                ":",
                self.id,
                len(self.underwritten_contracts),
                self.cash,
                self.operational,
            )

        self.make_reinsurance_claims(time)

        contracts_dissolved = self.mature_contracts(time)

        """effect payments from contracts"""
        for contract in self.underwritten_contracts:
            contract.check_payment_due(time)

        self.collect_process_evaluate_risks(time, contracts_dissolved)

        """adjust liquidity, borrow or invest"""
        # Not implemented


        self.market_permanency(time)

        self.roll_over(time)

        self.estimate_var()

    def collect_process_evaluate_risks(self, time, contracts_dissolved):
        if self.operational:

            """request risks to be considered for underwriting in the next period and collect those for this period"""
            new_nonproportional_risks, new_risks = self.get_newrisks_by_type()
            contracts_offered = len(new_risks)
            if isleconfig.verbose and contracts_offered < 2 * contracts_dissolved:
                print(
                    "Something wrong; agent {0:d} receives too few new contracts {1:d} <= {2:d}".format(
                        self.id, contracts_offered, 2 * contracts_dissolved
                    )
                )

            """deal with non-proportional risks first as they must evaluate each request separatly,
             then with proportional ones"""

            # Here the new reinrisks are organized by category.
            reinrisks_per_categ, number_reinrisks_categ = self.risks_reinrisks_organizer(
                new_nonproportional_risks
            )

            assert self.recursion_limit > 0
            for repetition in range(self.recursion_limit):
                # TODO: find an efficient way to stop the loop if there are no more risks to accept or if it is
                #  not accepting any more over several iterations.
                # Here we process all the new reinrisks in order to keep the portfolio as balanced as possible.
                former_reinrisks_per_categ = copy.copy(reinrisks_per_categ)
                [
                    reinrisks_per_categ,
                    not_accepted_reinrisks,
                ] = self.process_newrisks_reinsurer(
                    reinrisks_per_categ, number_reinrisks_categ, time
                )

                # QUERY: I moved this into the loop - was this correct?
                #  The loop only runs once in my tests, what needs tweaking to have firms not accept risks?
                self.simulation.return_reinrisks(not_accepted_reinrisks)

                if former_reinrisks_per_categ == reinrisks_per_categ:
                    # Stop condition implemented. Might solve the previous TODO.
                    break

            # QUERY: it's typically dangerous to compare floats with !=, is it okay in this case? Probably, since
            #  no arithmetic is done
            underwritten_risks = [
                {
                    "value": contract.value,
                    "category": contract.category,
                    "risk_factor": contract.risk_factor,
                    "deductible": contract.deductible,
                    "excess": contract.excess,
                    "insurancetype": contract.insurancetype,
                    "runtime": contract.runtime,
                }
                for contract in self.underwritten_contracts
                if contract.reinsurance_share != 1.0
            ]

            """obtain risk model evaluation (VaR) for underwriting decisions and for capacity specific decisions"""
            # TODO: Enable reinsurance shares other than 0.0 and 1.0
            expected_profit, acceptable_by_category, cash_left_by_categ, var_per_risk_per_categ, self.excess_capital = self.riskmodel.evaluate(
                underwritten_risks, self.cash
            )
            # TODO: resolve insurance reinsurance inconsistency (insurer underwrite after capacity decisions, reinsurers before).
            #  This is currently so because it minimizes the number of times we need to run self.riskmodel.evaluate().
            #  It would also be more consistent if excess capital would be updated at the end of the iteration.
            """handle adjusting capacity target and capacity"""
            max_var_by_categ = self.cash - self.excess_capital
            self.adjust_capacity_target(max_var_by_categ)
            actual_capacity = self.increase_capacity(time, max_var_by_categ)
            # TODO: make independent of insurer/reinsurer, but change this to different deductible values

            """handle capital market interactions: capital history, dividends"""
            self.cash_last_periods = [self.cash] + self.cash_last_periods[:3]
            self.adjust_dividends(time, actual_capacity)
            self.pay_dividends(time)

            """make underwriting decisions, category-wise"""
            growth_limit = max(
                50, 2 * len(self.underwritten_contracts) + contracts_dissolved
            )
            if sum(acceptable_by_category) > growth_limit:
                acceptable_by_category = np.asarray(acceptable_by_category).astype(
                    np.double
                )
                acceptable_by_category = (
                    acceptable_by_category * growth_limit / sum(acceptable_by_category)
                )
                acceptable_by_category = np.int64(np.round(acceptable_by_category))

            # Here the new risks are organized by category.
            [risks_per_categ, number_risks_categ] = self.risks_reinrisks_organizer(
                new_risks
            )

            for repetition in range(self.recursion_limit):
                # TODO: find an efficient way to stop the recursion if there are no more risks to accept or if it is not accepting any more over several iterations.
                former_risks_per_categ = copy.copy(risks_per_categ)
                # Here we process all the new risks in order to keep the portfolio as balanced as possible.
                risks_per_categ, not_accepted_risks = self.process_newrisks_insurer(
                    risks_per_categ,
                    number_risks_categ,
                    acceptable_by_category,
                    var_per_risk_per_categ,
                    cash_left_by_categ,
                    time,
                )
                # QUERY: As above, moved inside loop
                self.simulation.return_risks(not_accepted_risks)
                if (
                    former_risks_per_categ == risks_per_categ
                ):  # Stop condition implemented. Might solve the previous TODO.
                    break

            # print(self.id, " now has ", len(self.underwritten_contracts), " & returns ", len(not_accepted_risks))

    def enter_illiquidity(self, time):
        """Enter_illiquidity Method.
               Accepts arguments
                   time: Type integer. The current time.
               No return value.
           This method is called when a firm does not have enough cash to pay all its obligations. It is only called from
           the method self.effect_payments() which is called at the beginning of the self.iterate() method of this class.
           This method formalizes the bankruptcy through the method self.enter_bankruptcy()."""
        self.enter_bankruptcy(time)

    def enter_bankruptcy(self, time):
        """Enter_bankruptcy Method.
               Accepts arguments
                   time: Type integer. The current time.
               No return value.
           This method is used when a firm does not have enough cash to pay all its obligations. It is only called from
           the method self.enter_illiquidity() which is only called from the method self.effect_payments(). This method
           dissolves the firm through the method self.dissolve()."""
        self.dissolve(time, "record_bankruptcy")

    def market_exit(self, time):
        """Market_exit Method.
               Accepts arguments
                   time: Type integer. The current time.
               No return value.
           This method is called when a firms wants to leave the market because it feels that it has been underperforming
           for too many periods. It is only called from the method self.market_permanency() that it is run in the main iterate
           method of this class. It needs to be different from the method self.enter_bankruptcy() because in this case
           all the obligations can be paid. After paying all the obligations this method dissolves the firm through the
           method self.dissolve()."""
        due = [item for item in self.obligations]
        for obligation in due:
            self.pay(obligation)
        self.obligations = []
        self.dissolve(time, "record_market_exit")

    def dissolve(self, time, record):
        """Dissolve Method.
               Accepts arguments
                   time: Type integer. The current time.
                   record: Type string. This argument is a string that represents the kind of record that we want for
                   the dissolution of the firm.So far it can be either 'record_bankruptcy' or 'record_market_exit'.
               No return value.
           This method dissolves the firm. It is called from the methods self.enter_bankruptcy() and self.market_exit()
           of this class (metainsuranceorg.py). First of all it dissolves all the contracts currently held (those in self.underwritten_contracts).
           Next all the cash currently available is transferred to insurancesimulation.py through an obligation in the
           next iteration. Finally the type of dissolution is recorded and the operational state is set to false.
           Different class variables are reset during the process: self.risks_kept, self.reinrisks_kept, self.excess_capital
           and self.profits_losses."""
        for contract in self.underwritten_contracts:
            contract.dissolve(time)
        # removing (dissolving) all risks immediately after bankruptcy (may not be realistic,
        # they might instead be bought by another company)
        # TODO: implement buyouts
        self.simulation.return_risks(self.risks_kept)
        self.risks_kept = []
        self.reinrisks_kept = []
        obligation = {
            "amount": self.cash,
            "recipient": self.simulation,
            "due_time": time,
            "purpose": "Dissolution",
        }
        self.pay(
            obligation
        )  # This MUST be the last obligation before the dissolution of the firm.
        self.excess_capital = (
            0
        )  # Excess of capital is 0 after bankruptcy or market exit.
        self.profits_losses = (
            0
        )  # Profits and losses are 0 after bankruptcy or market exit.
        if self.operational:
            method_to_call = getattr(self.simulation, record)
            method_to_call()
        for category_reinsurance in self.category_reinsurance:
            if category_reinsurance is not None:
                category_reinsurance.dissolve(time)
        self.operational = False

    def receive_obligation(self, amount, recipient, due_time, purpose):
        """Method for receiving obligations that the firm will have to pay.
                    Accepts:
                        amount: Type integer, how much will be payed
                        recipient: Type Class instance, who will be payed
                        due_time: Type Integer, what time value they will be payed
                        purpose: Type string, why they are being payed
                    No return value
                    Adds obligation (Type DataDict) to list of obligations owed by the firm."""

        obligation = {
            "amount": amount,
            "recipient": recipient,
            "due_time": due_time,
            "purpose": purpose,
        }
        self.obligations.append(obligation)

    def effect_payments(self, time):"""Method for checking if any payments are due.
            Accepts:
                time: Type Integer
            No return value
            Method checks firms list of obligations to see if ay are due for this time, then pays them. If the firm
            does not have enough cash then it enters illiquity, leaves the market, and matures all contracts."""

        # TODO: don't really want to be reconstructing lists every time (unless the oblications are naturally sorted by
        #  time, in which case this could be done slightly better). Low priority, but something to consider
        due = [item for item in self.obligations if item["due_time"] <= time]
        self.obligations = [
            item for item in self.obligations if item["due_time"] > time
        ]
        # TODO: could this cause a firm to enter illiquidity if it has obligations to non-operational firms? Such
        #  firms can't recieve payment, so this possibly shouldn't happen.
        sum_due = sum([item["amount"] for item in due])
        if sum_due > self.cash:
            self.obligations += due
            self.enter_illiquidity(time)
            self.simulation.record_unrecovered_claims(sum_due - self.cash)
            # TODO: is this record of uncovered claims correct or should it be sum_due (since the company is
            #  impounded and self.cash will also not be paid out for quite some time)?
            # TODO: effect partial payment
        else:
            for obligation in due:
                self.pay(obligation)

    def pay(self, obligation):
        """Method to pay other class instances.
            Accepts:
                Obligation: Type DataDict
            No return value
            Method removes value payed from the agents cash and adds it to recipient agents cash."""
        amount = obligation["amount"]
        recipient = obligation["recipient"]
        purpose = obligation["purpose"]
        if self.get_operational() and recipient.get_operational():
            self.cash -= amount
            if purpose is not "dividend":
                self.profits_losses -= amount
            recipient.receive(amount)

    def receive(self, amount):
        """Method to accept cash payments.
            Accepts:
                amount: Type Integer
            No return value"""
        self.cash += amount
        self.profits_losses += amount

    def pay_dividends(self, time):
        """Method to receive dividend obligation.
                    Accepts:
                        time: Type integer
                    No return value
                    If firm has positive profits will pay percentage of them as dividends. Currently pays to simulation."""

        self.receive_obligation(self.per_period_dividend, self.owner, time, "dividend")

    def obtain_yield(self, time):
        """Method to obtain intereset on cash reserves
        Accepts:
            time: Type integer
            No return value"""
        amount = self.cash * self.interest_rate
        # TODO: agent should not award her own interest.
        #  This interest rate should be taken from self.simulation with a getter method
        self.simulation.receive_obligation(amount, self, time, "yields")

    def mature_contracts(self, time):
        """Method to mature contracts that have expired
        Accepts:
            time: Type integer
            Returns:
                number of contracts maturing: Type integer"""
        if isleconfig.verbose:
            print("Number of underwritten contracts ", len(self.underwritten_contracts))
        maturing = [
            contract
            for contract in self.underwritten_contracts
            if contract.expiration <= time
        ]
        for contract in maturing:
            self.underwritten_contracts.remove(contract)
            contract.mature(time)
        return len(maturing)

    def get_cash(self):
        """Method to return agents cash. Only used to calculate total sum of capital to recalculate market premium
            each iteration.
           No accepted values.
           No return values."""
        return self.cash

    def get_excess_capital(self):
        """Method to get agents excess capital. Only used for saving data. Called by simulation.
            No Accepted values.
            Returns agents excess capital"""
        return self.excess_capital

    def logme(self):
        self.log("cash", self.cash)
        self.log("underwritten_contracts", self.underwritten_contracts)
        self.log("operational", self.operational)

    def log(self, *args):
        raise NotImplementedError(
            "The log method should have been overridden by the subclass"
        )

    def number_underwritten_contracts(self):
        return len(self.underwritten_contracts)

    def get_underwritten_contracts(self):
        return self.underwritten_contracts

    def get_profitslosses(self):
        """Method to get agents profit or loss. Only used for saving data. Called by simulation.
            No Accepted values.
            Returns agents profits/losses"""
        return self.profits_losses

    def get_operational(self):
        """Method to return boolean of if agent is operational. Only used as check for payments.
            No accepted values
            Returns Boolean"""
        return self.operational

    def get_pointer(self):
        """Method to get pointer. Returns self so renduant? Called only by resume.py"""
        return self

    def estimate_var(self):
        """Method to estimate Value at Risk.
            No Accepted arguments.
            No return values
           Calculates value at risk per category and overall, based on underwritten contracts initial value at risk.
           Assigns it to agent instance. Called at the end of each agents iteration cycle."""
        self.counter_category = np.zeros(self.simulation_no_risk_categories)
        self.var_category = np.zeros(self.simulation_no_risk_categories)

        self.var_counter = 0
        self.var_counter_per_risk = 0
        self.var_sum = 0

        if self.operational:

            for contract in self.underwritten_contracts:
                self.counter_category[contract.category] += 1
                self.var_category[contract.category] += contract.initial_VaR

            for category in range(len(self.counter_category)):
                self.var_counter += (
                    self.counter_category[category]
                    * self.riskmodel.inaccuracy[category]
                )
                self.var_sum += self.var_category[category]

            if sum(self.counter_category) != 0:
                self.var_counter_per_risk = self.var_counter / sum(
                    self.counter_category
                )
            else:
                self.var_counter_per_risk = 0

    def get_newrisks_by_type(self):
        """Method for soliciting new risks from insurance simulation then organising them based if non-proportional
            or not.
            No accepted Values.
            Returns:
                new_non_proportional_risks: Type list of DataDicts.
                new_risks: Type list of DataDicts."""
        new_risks = []
        if self.is_insurer:
            new_risks += self.simulation.solicit_insurance_requests(
                self.id, self.cash, self
            )
        if self.is_reinsurer:
            new_risks += self.simulation.solicit_reinsurance_requests(
                self.id, self.cash, self
            )

        new_nonproportional_risks = [
            risk
            for risk in new_risks
            if risk.get("insurancetype") == "excess-of-loss"
            and risk["owner"] is not self
        ]
        new_risks = [
            risk
            for risk in new_risks
            if risk.get("insurancetype") in ["proportional", None]
            and risk["owner"] is not self
        ]
        return new_nonproportional_risks, new_risks


    def increase_capacity(self, time, var_by_category):
        raise NotImplementedError(
            "Method is not implemented in MetaInsuranceOrg, just in inheriting InsuranceFirm instances"
        )

    def adjust_dividends(self, time, actual_capacity):
        raise NotImplementedError(
            "Method not implemented. adjust_dividends method should be implemented in inheriting classes"
        )

    def adjust_capacity_target(self, time):
        raise NotImplementedError(
            "Method not implemented. adjust_capacity_target method should be implemented in inheriting classes"
        )

    def risks_reinrisks_organizer(self, new_risks):  #This method organizes the new risks received by the insurer (or reinsurer)
        """This method organizes the new risks received by the insurer (or reinsurer) by category.
                    Accepts:
                        new_risks: Type list of DataDicts
                    Returns:
                        risks_per_catgegory: Type list of categories, each contains risks originating from that category.
                        number_risks_categ: Type list, elements are integers of total risks in each category"""
        risks_per_categ = [[] for x in range(self.simulation_parameters["no_categories"])]      #This method organizes the new risks received by the insurer (or reinsurer) by category in the nested list "risks_per_categ".
        number_risks_categ = [[] for x in range(self.simulation_parameters["no_categories"])]   #This method also counts the new risks received by the insurer (or reinsurer) by category in the list "number_risks_categ".

        for categ_id in range(self.simulation_parameters["no_categories"]):
            risks_per_categ[categ_id] = [
                risk for risk in new_risks if risk["category"] == categ_id
            ]
            number_risks_categ[categ_id] = len(risks_per_categ[categ_id])

        # The method returns both risks_per_categ and number_risks_categ.
        return risks_per_categ, number_risks_categ

    def balanced_portfolio(self, risk, cash_left_by_categ, var_per_risk):
        """This method decides whether the portfolio is balanced enough to accept a new risk or not. If it is balanced
            enough return True otherwise False. This method also returns the cash available per category independently
            the risk is accepted or not.
           Accepts:
                risk: Type DataDict
                cash_left_by_category: Type List, contains list of available cash per category
                var_per_risk: Type list of integers contains VaR for each category defined in getPPF from riskmodel.py
            Returns:
                Boolean
                cash_left_by_categ: Type list of integers"""

        # Compute the cash already reserved by category
        cash_reserved_by_categ = self.cash - cash_left_by_categ

        _, std_pre = get_mean_std(tuple(cash_reserved_by_categ))

        # For some reason just recreating the array is faster than copying it
        # cash_reserved_by_categ_store = np.copy(cash_reserved_by_categ)
        cash_reserved_by_categ_store = np.array(cash_reserved_by_categ)

        if risk.get("insurancetype") == "excess-of-loss":
            percentage_value_at_risk = self.riskmodel.get_ppf(
                categ_id=risk["category"], tail_size=self.riskmodel.var_tail_prob
            )
            expected_damage = (
                percentage_value_at_risk
                * risk["value"]
                * risk["risk_factor"]
                * self.riskmodel.inaccuracy[risk["category"]]
            )
            expected_claim = (
                min(expected_damage, risk["value"] * risk["excess_fraction"])
                - risk["value"] * risk["deductible_fraction"]
            )

            # record liquidity requirement and apply margin of safety for liquidity requirement

            # Compute how the cash reserved by category would change if the new reinsurance risk was accepted
            cash_reserved_by_categ_store[risk["category"]] += (
                expected_claim * self.riskmodel.margin_of_safety
            )

        else:
            # Compute how the cash reserved by category would change if the new insurance risk was accepted
            cash_reserved_by_categ_store[risk["category"]] += var_per_risk[
                risk["category"]
            ]

        # Compute the mean, std of the cash reserved by category after the new risk of reinrisk is accepted
        mean, std_post = get_mean_std(tuple(cash_reserved_by_categ_store))

        total_cash_reserved_by_categ_post = sum(cash_reserved_by_categ_store)

        # Doing a < b*c is about 10% faster than a/c < b
        if (std_post * total_cash_reserved_by_categ_post) <= (
            self.balance_ratio * mean * self.cash
        ) or std_post < std_pre:
            # The new risk is accepted if the standard deviation is reduced or the cash reserved by category is very
            # well balanced. (std_post) <= (self.balance_ratio * mean)
            # The balance condition is not taken into account if the cash reserve is far away from the limit.
            # (total_cash_employed_by_categ_post/self.cash <<< 1)
            cash_left_by_categ = self.cash - cash_reserved_by_categ_store

            return True, cash_left_by_categ
        else:
            cash_left_by_categ = self.cash - cash_reserved_by_categ

            return False, cash_left_by_categ

    def process_newrisks_reinsurer(
        self, reinrisks_per_categ, number_reinrisks_categ, time
    ):
        """Method to decide if new risks are underwritten for the reinsurance firm.
            Accepts:
                reinrisks_per_categ: Type List of lists containing new reinsurance risks.
                number_reinrisks_per_categ: Type List of integers, contains number of new reinsurance risks per category.
                time: Type integer
            No return values.
           This method processes one by one the reinrisks contained in reinrisks_per_categ in order to decide whether
           they should be underwritten or not. It is done in this way to maintain the portfolio as balanced as possible.
           For that reason we process risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... and so forth. If
           risks are accepted then a contract is written."""

        for iterion in range(max(number_reinrisks_categ)):
            for categ_id in range(self.simulation_parameters["no_categories"]):   #Here we take only one risk per category at a time to achieve risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... if possible.
                if iterion < number_reinrisks_categ[categ_id] and reinrisks_per_categ[categ_id][iterion] is not None:
                    risk_to_insure = reinrisks_per_categ[categ_id][iterion]
                    underwritten_risks = [
                        {
                            "value": contract.value,
                            "category": contract.category,
                            "risk_factor": contract.risk_factor,
                            "deductible": contract.deductible,
                            "excess": contract.excess,
                            "insurancetype": contract.insurancetype,
                            "runtime_left": (contract.expiration - time),
                        }
                        for contract in self.underwritten_contracts
                        if contract.insurancetype == "excess-of-loss"
                    ]
                    accept, cash_left_by_categ, var_this_risk, self.excess_capital = self.riskmodel.evaluate(
                        underwritten_risks, self.cash, risk_to_insure
                    )
                    # TODO: change riskmodel.evaluate() to accept new risk to be evaluated and
                    #  to account for existing non-proportional risks correctly -> DONE.
                    if accept:
                        # TODO: rename this to per_value_premium in insurancecontract.py to avoid confusion
                        per_value_reinsurance_premium = (
                            self.np_reinsurance_premium_share
                            * risk_to_insure["periodized_total_premium"]
                            * risk_to_insure["runtime"]
                            * (
                                self.simulation.get_market_reinpremium()
                                / self.simulation.get_market_premium()
                            )
                            / risk_to_insure["value"]
                        )
                        # Here it is check whether the portfolio is balanced or not if the reinrisk
                        # (risk_to_insure) is underwritten. Return True if it is balanced. False otherwise.
                        condition, cash_left_by_categ = self.balanced_portfolio(
                            risk_to_insure, cash_left_by_categ, None
                        )

                        if condition:
                            contract = ReinsuranceContract(
                                self,
                                risk_to_insure,
                                time,
                                per_value_reinsurance_premium,
                                risk_to_insure["runtime"],
                                self.default_contract_payment_period,
                                expire_immediately=self.simulation_parameters[
                                    "expire_immediately"
                                ],
                                initial_var=var_this_risk,
                                insurancetype=risk_to_insure["insurancetype"],
                            )  # TODO: implement excess of loss for reinsurance contracts
                            self.underwritten_contracts.append(contract)
                            self.cash_left_by_categ = cash_left_by_categ
                            reinrisks_per_categ[categ_id][iterion] = None

        not_accepted_reinrisks = []
        for categ_id in range(self.simulation_parameters["no_categories"]):
            for reinrisk in reinrisks_per_categ[categ_id]:
                if reinrisk is not None:
                    not_accepted_reinrisks.append(reinrisk)

        return reinrisks_per_categ, not_accepted_reinrisks

    def process_newrisks_insurer(
        self,
        risks_per_categ,
        number_risks_categ,
        acceptable_by_category,
        var_per_risk_per_categ,
        cash_left_by_categ,
        time,
    ):
        """Method to decide if new risks are underwritten for the insurance firm.
            Accepts:
                risks_per_categ: Type List of lists containing new risks.
                number_risks_categ: Type List of integers, contains number of new risks per category.
                acceptable_per_category:
                var_per_risk_per_categ: Type list of integers contains VaR for each category defined in getPPF.
                cash_left_by_categ:  Type List, contains list of available cash per category
                time: Type integer.
            Returns:
                risks_per_categ: Type list of list, same as above however with None where contracts were accepted.
                not_accepted_risks: Type List of DataDicts
        This method processes one by one the reinrisks contained in reinrisks_per_categ in order to decide whether
        they should be underwritten or not. It is done in this way to maintain the portfolio as balanced as possible.
        For that reason we process risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... and so forth. If
        risks are accepted then a contract is written."""
_cached_rvs = self.contract_runtime_dist.rvs()
        for risk_index in range(max(number_risks_categ)):
            for categ_id in range(len(acceptable_by_category)):
                if (
                    risk_index < number_risks_categ[categ_id]
                    and acceptable_by_category[categ_id] > 0
                    and risks_per_categ[categ_id][risk_index] is not None
                ):
                    risk_to_insure = risks_per_categ[categ_id][risk_index]
                    if (
                        "contract" in risk_to_insure
                        and risk_to_insure["contract"].expiration > time
                    ):
                        # In this case the risk being inspected already has a contract, so we are deciding whether to
                        # give reinsurance for it # QUERY: is this correct?
                        [condition, cash_left_by_categ] = self.balanced_portfolio(
                            risk_to_insure, cash_left_by_categ, None
                        )
                        # Here we check whether the portfolio is balanced or not if the reinrisk (risk_to_insure) is
                        # underwritten. Return True if it is balanced. False otherwise.
                        if condition:
                            contract = ReinsuranceContract(
                                self,
                                risk_to_insure,
                                time,
                                self.simulation.get_reinsurance_market_premium(),
                                risk_to_insure["expiration"] - time,
                                self.default_contract_payment_period,
                                expire_immediately=self.simulation_parameters[
                                    "expire_immediately"
                                ],
                            )
                            self.underwritten_contracts.append(contract)
                            self.cash_left_by_categ = cash_left_by_categ
                            risks_per_categ[categ_id][risk_index] = None
                            # TODO: move this to insurancecontract (ca. line 14) -> DONE
                            # TODO: do not write into other object's properties, use setter -> DONE
                    else:
                        [condition, cash_left_by_categ] = self.balanced_portfolio(
                            risk_to_insure, cash_left_by_categ, var_per_risk_per_categ
                        )
                        # In this case there is no contact currently associated with the risk, so we decide whether
                        # to insure it
                        # Here it is check whether the portfolio is balanced or not if the risk (risk_to_insure) is
                        # underwritten. Return True if it is balanced. False otherwise.
                        if condition:
                            contract = InsuranceContract(
                                self,
                                risk_to_insure,
                                time,
                                self.simulation.get_market_premium(),
                                _cached_rvs,
                                self.default_contract_payment_period,
                                expire_immediately=self.simulation_parameters[
                                    "expire_immediately"
                                ],
                                initial_var=var_per_risk_per_categ[categ_id],
                            )
                            self.underwritten_contracts.append(contract)
                            self.cash_left_by_categ = cash_left_by_categ
                            risks_per_categ[categ_id][risk_index] = None
                    acceptable_by_category[categ_id] -= 1
                    # TODO: allow different values per risk (i.e. sum over value (and reinsurance_share) or
                    #  exposure instead of counting)

        not_accepted_risks = []
        for categ_id in range(len(acceptable_by_category)):
            for risk in risks_per_categ[categ_id]:
                if risk is not None:
                    not_accepted_risks.append(risk)

        return risks_per_categ, not_accepted_risks

    def market_permanency(self, time):
        """Method determining if firm stays in market.
            Accepts:
                Time: Type Integer
            No return values. This method determines whether an insurer or reinsurer stays in the market.
        If it has very few risks underwritten or too much cash left for TOO LONG it eventually leaves the market.
        If it has very few risks underwritten it cannot balance the portfolio so it makes sense to leave the market."""
        if not self.simulation_parameters["market_permanency_off"]:

            cash_left_by_categ = np.asarray(self.cash_left_by_categ)

            avg_cash_left = get_mean(cash_left_by_categ)

            if (
                self.cash < self.simulation_parameters["cash_permanency_limit"]
            ):  # If their level of cash is so low that they cannot underwrite anything they also leave the market.
                self.market_exit(time)
            else:
                if self.is_insurer:

                    if (
                        len(self.underwritten_contracts)
                        < self.simulation_parameters[
                            "insurance_permanency_contracts_limit"
                        ]
                        or avg_cash_left / self.cash
                        > self.simulation_parameters["insurance_permanency_ratio_limit"]
                    ):
                        # Insurers leave the market if they have contracts under the limit or an excess capital
                        # over the limit for too long.
                        self.market_permanency_counter += 1
                    else:
                        self.market_permanency_counter = 0
                    if (
                        self.market_permanency_counter
                        >= self.simulation_parameters[
                            "insurance_permanency_time_constraint"
                        ]
                    ):
                        # Here we determine how much is too long.
                        self.market_exit(time)

                if self.is_reinsurer:

                    if (
                        len(self.underwritten_contracts)
                        < self.simulation_parameters[
                            "reinsurance_permanency_contracts_limit"
                        ]
                        or avg_cash_left / self.cash
                        > self.simulation_parameters[
                            "reinsurance_permanency_ratio_limit"
                        ]
                    ):
                        # Reinsurers leave the market if they have contracts under the limit or an excess capital
                        # over the limit for too long.

                        self.market_permanency_counter += 1
                        # Insurers and reinsurers potentially have different reasons to leave the market.
                        # That's why the code is duplicated here.
                    else:
                        self.market_permanency_counter = 0

                    if (
                        self.market_permanency_counter
                        >= self.simulation_parameters[
                            "reinsurance_permanency_time_constraint"
                        ]
                    ):
                        # Here we determine how much is too long.
                        self.market_exit(time)

    def register_claim(self, claim):
        """Method to register claims.
            Accepts:
                claim: Type Integer, value of claim.
            No return values.
        This method records in insurancesimulation.py every claim made. It is called either from insurancecontract.py
        or reinsurancecontract.py respectively."""
        self.simulation.record_claims(claim)

    def reset_pl(self):
        """Reset_pl Method.
               Accepts no arguments:
               No return value.
           Reset the profits and losses variable of each firm at the beginning of every iteration. It has to be run in
           insurancesimulation.py at the beginning of the iterate method"""
        self.profits_losses = 0

    def roll_over(self, time):
        """Roll_over Method.
               Accepts arguments
                   time: Type integer. The current time.               No return value.
               No return value.
            This method tries to roll over the insurance and reinsurance contracts expiring in the next iteration. In
            the case of insurance contracts it assumes that it can only retain a fraction of contracts inferior to the
            retention rate. The contracts that cannot be retained are sent back to insurancesimulation.py. The rest are
            kept and evaluated the next iteration. For reinsurancecontracts is exactly the same with the difference that
            there is no need to return the contracts not rolled over to insurancesimulation.py, since reinsurance risks
            are created and destroyed every iteration. The main reason to implemented this method is to avoid a lack of
            coverage that appears, if contracts are allowed to mature and are evaluated again the next iteration."""

        maturing_next = [
            contract
            for contract in self.underwritten_contracts
            if contract.expiration == time + 1
        ]
        # QUERY: Is it true to say that no firm underwrites both insurance and reinsurance?
        # Generate all the rvs at the start
        if maturing_next:
            uniform_rvs = np.nditer(np.random.uniform(size=len(maturing_next)))
            if self.is_insurer:
                for contract in maturing_next:
                    contract.roll_over_flag = 1
                    if (
                        next(uniform_rvs)
                        > self.simulation_parameters["insurance_retention"]
                    ):
                        self.simulation.return_risks(
                            [contract.risk_data]
                        )  # TODO: This is not a retention, so the roll_over_flag might be confusing in this case
                    else:
                        self.risks_kept.append(contract.risk_data)

            if self.is_reinsurer:
                for reincontract in maturing_next:
                    if reincontract.property_holder.operational:
                        reincontract.roll_over_flag = 1
                        reinrisk = reincontract.property_holder.create_reinrisk(
                            time, reincontract.category
                        )
                        if (
                            next(uniform_rvs)
                            < self.simulation_parameters["reinsurance_retention"]
                        ):
                            if reinrisk is not None:
                                self.reinrisks_kept.append(reinrisk)

    def make_reinsurance_claims(self, time):
        raise NotImplementedError(
            "MetaInsuranceOrg does not implement make_reinsurance_claims, "
            "it should have been overridden"
        )
