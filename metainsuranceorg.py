import math
import functools
from itertools import cycle, islice, chain

import numpy as np
import scipy.stats

import isleconfig
import insurancecontract
import reinsurancecontract
import riskmodel
from genericclasses import (
    GenericAgent,
    RiskProperties,
    AgentProperties,
    Obligation,
    ReinsuranceProfile,
)

from typing import (
    Optional,
    Tuple,
    Sequence,
    Mapping,
    MutableSequence,
    Iterable,
    Callable,
    Any,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurancesimulation import InsuranceSimulation
    from metainsurancecontract import MetaInsuranceContract
    from reinsurancecontract import ReinsuranceContract


def roundrobin(iterables: Sequence[Iterable]) -> Iterable:
    """roundrobin(['ABC', 'D', 'EF']) --> A D E B F C"""
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts: Iterable[Callable[[], Any]] = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_fun in nexts:
                yield next_fun()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def get_mean(x: Sequence[float]) -> float:
    """
    Returns the mean of a list
    Args:
        x: an iterable of numerics

    Returns:
        the mean of x
    """
    return sum(x) / len(x)


# A quick check tells me that we don't need a very large cache for this, as it only tends to repeat a couple of times.
@functools.lru_cache(maxsize=16)
def get_mean_std(x: Tuple[float, ...]) -> Tuple[float, float]:
    # At the moment this is always called with a no_category length array
    # I have tested the numpy versions of this, they are slower for small arrays but much, much faster for large ones
    # If we ever let no_category be much larger, might want to use np for this bit
    m = get_mean(x)
    std = math.sqrt(sum((val - m) ** 2 for val in x)) / len(x)
    return m, std


class MetaInsuranceOrg(GenericAgent):
    def __init__(self, simulation_parameters: Mapping, agent_parameters: AgentProperties):
        """Constructor method.
                    Accepts:
                        Simulation_parameters: Type DataDict
                        agent_parameters:   Type DataDict
                    Constructor creates general instance of an insurance company which is inherited by the reinsurance
                     and insurance firm classes. Initialises all necessary values provided by config file."""
        super().__init__()
        self.simulation: "InsuranceSimulation" = simulation_parameters["simulation"]
        self.simulation_parameters: Mapping = simulation_parameters
        self.contract_runtime_dist = scipy.stats.randint(simulation_parameters["mean_contract_runtime"]
                                                         - simulation_parameters["contract_runtime_halfspread"],
                                                         simulation_parameters["mean_contract_runtime"]
                                                         + simulation_parameters["contract_runtime_halfspread"] + 1)
        self.default_contract_payment_period: int = simulation_parameters["default_contract_payment_period"]
        self.id = agent_parameters.id
        self.cash = agent_parameters.initial_cash
        self.capacity_target = self.cash * 0.9
        self.capacity_target_decrement_threshold = (agent_parameters.capacity_target_decrement_threshold)
        self.capacity_target_increment_threshold = (agent_parameters.capacity_target_increment_threshold)
        self.capacity_target_decrement_factor = (agent_parameters.capacity_target_decrement_factor)
        self.capacity_target_increment_factor = (agent_parameters.capacity_target_increment_factor)
        self.excess_capital = self.cash
        self.premium = agent_parameters.norm_premium
        self.profit_target = agent_parameters.profit_target
        self.acceptance_threshold = agent_parameters.initial_acceptance_threshold  # 0.5
        self.acceptance_threshold_friction = (agent_parameters.acceptance_threshold_friction)  # 0.9 #1.0 to switch off
        self.interest_rate = agent_parameters.interest_rate
        self.reinsurance_limit = agent_parameters.reinsurance_limit
        self.simulation_no_risk_categories = simulation_parameters["no_categories"]
        self.simulation_reinsurance_type = simulation_parameters["simulation_reinsurance_type"]
        self.dividend_share_of_profits = simulation_parameters["dividend_share_of_profits"]

        # If the firm goes bankrupt then by default any further payments should be made to the simulation
        self.creditor = self.simulation
        self.owner = self.simulation  # TODO: Make this into agent_parameter value?
        self.per_period_dividend = 0
        self.cash_last_periods = list(np.zeros(12, dtype=int) * self.cash)

        rm_config = agent_parameters.riskmodel_config

        """Here we modify the margin of safety depending on the number of risks models available in the market.
           When is 0 all risk models have the same margin of safety. The reason for doing this is that with more risk
           models the firms tend to be closer to the max capacity"""
        margin_of_safety_correction = (rm_config["margin_of_safety"]+ (simulation_parameters["no_riskmodels"] - 1) * simulation_parameters["margin_increase"])

        self.max_inaccuracy = rm_config["inaccuracy_by_categ"]
        self.min_inaccuracy = self.max_inaccuracy * isleconfig.simulation_parameters["scale_inaccuracy"] + \
                              np.ones(len(self.max_inaccuracy)) * (1 - isleconfig.simulation_parameters["scale_inaccuracy"])

        self.riskmodel: riskmodel.RiskModel = riskmodel.RiskModel(
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
            inaccuracy=self.max_inaccuracy,)

        # Set up the reinsurance profile
        self.reinsurance_profile = ReinsuranceProfile(self.riskmodel)

        if self.simulation_reinsurance_type == "non-proportional":
            if agent_parameters.non_proportional_reinsurance_level is not None:
                self.np_reinsurance_deductible_fraction = (
                    agent_parameters.non_proportional_reinsurance_level
                )
            else:
                self.np_reinsurance_deductible_fraction = simulation_parameters[
                    "default_non-proportional_reinsurance_deductible"
                ]
            self.np_reinsurance_limit_fraction = simulation_parameters[
                "default_non-proportional_reinsurance_excess"
            ]
            self.np_reinsurance_premium_share = simulation_parameters[
                "default_non-proportional_reinsurance_premium_share"
            ]
        self.underwritten_contracts: MutableSequence["MetaInsuranceContract"] = []
        self.is_insurer = True
        self.is_reinsurer = False
        self.warning = False
        self.age = 0

        """set up risk value estimate variables"""
        self.var_counter = 0                # sum over risk model inaccuracies for all contracts
        self.var_counter_per_risk = 0       # average risk model inaccuracy across contracts
        self.var_sum = 0                    # sum over initial VaR for all contracts
        self.var_sum_last_periods = list(np.zeros((12, 4), dtype=int))
        self.reinsurance_history = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.counter_category = np.zeros(self.simulation_no_risk_categories)    # var_counter disaggregated by category
        self.var_category = np.zeros(self.simulation_no_risk_categories)        # var_sum disaggregated by category
        self.naccep = []
        self.risks_kept = []
        self.reinrisks_kept = []
        self.balance_ratio = simulation_parameters['insurers_balance_ratio']
        self.recursion_limit = simulation_parameters['insurers_recursion_limit']
        self.cash_left_by_categ = [self.cash for i in range(self.simulation_parameters["no_categories"])]
        self.market_permanency_counter = 0
        # TODO: make this into a dict
        self.underwritten_risk_characterisation: MutableSequence[Tuple[float, float, int, float]] = [
            (None, None, None, None)
            for _ in range(self.simulation_parameters["no_categories"])
        ]
        # The share of all risks that this firm holds. Gets updated every timestep
        self.risk_share = 0

    def iterate(self, time: int):
        """Method that iterates each firm by one time step.
                    Accepts:
                        Time: Type Integer
                    No return value
                    For each time step this method obtains every firms interest payments, pays obligations, claim
                    reinsurance, matures necessary contracts. Check condition for operational firms (as never removed)
                    so only operational firms receive new risks to evaluate, pay dividends, adjust capacity."""

        """Obtain interest generated by cash"""
        self.simulation.bank.award_interest(self, self.cash)
        self.age += 1

        """realize due payments"""
        self._effect_payments(time)
        if isleconfig.verbose:
            print(time, ":", self.id, len(self.underwritten_contracts), self.cash, self.operational)

        self.make_reinsurance_claims(time)

        contracts_dissolved = self.mature_contracts(time)

        """effect payments from contracts"""
        for contract in self.underwritten_contracts:
            contract.check_payment_due(time)

        """Check what proportion of the risk market we hold and then update the riskmodel accordingly"""
        self.update_risk_share()
        self.adjust_riskmodel_inaccuracy()

        if self.operational:
            # Firms submit cash and var data for regulation every 12 iterations
            if time % 12 == 0 and isleconfig.enforce_regulations is True:
                self.submit_regulator_report(time)
                if self.operational is False:   # If not enough average cash then firm is closed and so no underwriting.
                    return

        """Collect and process new risks"""
        self.collect_process_evaluate_risks(time, contracts_dissolved)

        """adjust liquidity, borrow or invest"""
        # Not implemented

        if self.operational and not self.warning:
            self.market_permanency(time)

            self.roll_over(time)

        self.estimate_var()

    def collect_process_evaluate_risks(self, time: int, contracts_dissolved: int):
        if self.operational:
            self.update_risk_characterisations()
            for categ in range(len(self.counter_category)):
                value = self.underwritten_risk_characterisation[categ][0]
                self.reinsurance_profile.update_value(value, categ)

            # Only get risks if firm not issued warning (breaks otherwise)
            if not self.warning:
                """request risks to be considered for underwriting in the next period and collect those for this period"""
                new_nonproportional_risks, new_risks = self.get_newrisks_by_type()
                contracts_offered = len(new_risks)
                if isleconfig.verbose and contracts_offered < 2 * contracts_dissolved:
                    print(f"Something wrong; agent {self.id} receives too few new contracts {contracts_offered} "
                         f"<= {2 * contracts_dissolved}")

                """deal with non-proportional risks first as they must evaluate each request separately,
                 then with proportional ones"""

                # Here the new reinrisks are organized by category.
                reinrisks_per_categ = self.risks_reinrisks_organizer(new_nonproportional_risks)

                assert self.recursion_limit > 0
                for repetition in range(self.recursion_limit):
                    # TODO: find an efficient way to stop the loop if there are no more risks to accept or if it is
                    #  not accepting any more over several iterations.
                    # Here we process all the new reinrisks in order to keep the portfolio as balanced as possible.
                    has_accepted_risks, not_accepted_reinrisks = self.process_newrisks_reinsurer(
                        reinrisks_per_categ, time)

                    #  The loop only runs once in my tests, what needs tweaking to have firms not accept risks?
                    reinrisks_per_categ = not_accepted_reinrisks
                    if not has_accepted_risks:
                        # Stop condition implemented. Might solve the previous TODO.
                        break
                self.simulation.return_reinrisks(list(chain.from_iterable(not_accepted_reinrisks)))

            underwritten_risks = [RiskProperties(
                                owner=self,
                                value=contract.value,
                                category=contract.category,
                                risk_factor=contract.risk_factor,
                                deductible=contract.deductible,
                                limit=contract.limit,
                                insurancetype=contract.insurancetype,
                                runtime=contract.runtime)
                                for contract in self.underwritten_contracts
                                if contract.reinsurance_share != 1.0]

            """obtain risk model evaluation (VaR) for underwriting decisions and for capacity specific decisions"""
            # TODO: Enable reinsurance shares other than 0.0 and 1.0
            [_, acceptable_by_category, cash_left_by_categ, var_per_risk_per_categ, self.excess_capital] = self.riskmodel.evaluate(underwritten_risks, self.cash)
            # TODO: resolve insurance reinsurance inconsistency (insurer underwrite after capacity decisions,
            #  reinsurers before).

            #  This is currently so because it minimizes the number of times we need to run self.riskmodel.evaluate().
            #  It would also be more consistent if excess capital would be updated at the end of the iteration.
            """handle adjusting capacity target and capacity"""
            max_var_by_categ = self.cash - self.excess_capital
            self.adjust_capacity_target(max_var_by_categ)

            self.update_risk_characterisations()

            actual_capacity = self.increase_capacity(time, max_var_by_categ)
            # TODO: make independent of insurer/reinsurer, but change this to different deductible values

            """handle capital market interactions: capital history, dividends"""
            self.cash_last_periods = np.roll(self.cash_last_periods, -1)
            self.cash_last_periods[-1] = self.cash
            self.adjust_dividends(time, actual_capacity)
            self.pay_dividends(time)

            # Firms only decide to underwrite if not issued a warning
            if not self.warning:
                """make underwriting decisions, category-wise"""
                growth_limit = max(50, 2 * len(self.underwritten_contracts) + contracts_dissolved)
                if sum(acceptable_by_category) > growth_limit:
                    acceptable_by_category = np.asarray(acceptable_by_category).astype(np.double)
                    acceptable_by_category = acceptable_by_category * growth_limit / sum(acceptable_by_category)
                    acceptable_by_category = np.int64(np.round(acceptable_by_category))

                # Here the new risks are organized by category.
                risks_per_categ = self.risks_reinrisks_organizer(new_risks)
                if risks_per_categ != [[] for _ in range(self.simulation_no_risk_categories)]:
                    for repetition in range(self.recursion_limit):
                        # Here we process all the new risks in order to keep the portfolio as balanced as possible.
                        has_accepted_risks, not_accepted_risks = self.process_newrisks_insurer(
                            risks_per_categ,
                            acceptable_by_category,
                            var_per_risk_per_categ,
                            cash_left_by_categ,
                            time)
                        risks_per_categ = not_accepted_risks
                        if not has_accepted_risks:
                            break
                    self.simulation.return_risks(list(chain.from_iterable(not_accepted_risks)))
            self.update_risk_characterisations()

    def enter_illiquidity(self, time: int):
        """Enter_illiquidity Method.
               Accepts arguments
                   time: Type integer. The current time.
                   sum_due: the outstanding sum that the firm couldn't pay
               No return value.
           This method is called when a firm does not have enough cash to pay all its obligations. It is only called from
           the method self.effect_payments() which is called at the beginning of the self.iterate() method of this class.
           This method formalizes the bankruptcy through the method self.enter_bankruptcy()."""
        self.enter_bankruptcy(time)

    def enter_bankruptcy(self, time: int):
        """Enter_bankruptcy Method.
               Accepts arguments
                   time: Type integer. The current time.
               No return value.
           This method is used when a firm does not have enough cash to pay all its obligations. It is only called from
           the method self.enter_illiquidity() which is only called from the method self._effect_payments(). This method
           dissolves the firm through the method self.dissolve()."""
        if isleconfig.buy_bankruptcies:
            if self.is_insurer and self.operational:
                self.simulation.add_firm_to_be_sold(self, time, "record_bankruptcy")
                self.operational = False
            elif self.is_reinsurer and self.operational:
                self.simulation.add_firm_to_be_sold(self, time, "record_bankruptcy")
                self.operational = False
            else:
                self.dissolve(time, 'record_bankruptcy')
        else:
            self.dissolve(time, 'record_bankruptcy')

    def market_exit(self, time):
        """Market_exit Method.
               Accepts arguments
                   time: Type integer. The current time.
               No return value.
           This method is called when a firms wants to leave the market because it feels that it has been
           underperforming for too many periods. It is only called from the method self.market_permanency() that it is
           run in the main iterate method of this class. It needs to be different from the method
           self.enter_bankruptcy() because in this case all the obligations can be paid. After paying all the
           obligations this method dissolves the firm through the method self.dissolve()."""
        due = [item for item in self.obligations]
        sum_due = sum([item.amount for item in due])
        if sum_due > self.cash:
            self.enter_bankruptcy(time)
            print("Dissolved due to market exit")
        for obligation in due:
            self._pay(obligation)
        self.obligations = []
        self.dissolve(time, 'record_market_exit')
        for contract in self.underwritten_contracts:
            contract.mature(time)
        self.underwritten_contracts = []

    def dissolve(self, time, record):
        """Dissolve Method.
               Accepts arguments
                   time: Type integer. The current time.
                   record: Type string. This argument is a string that represents the kind of record that we want for
                   the dissolution of the firm.So far it can be either 'record_bankruptcy' or 'record_market_exit'.
               No return value.
           This method dissolves the firm. It is called from the methods self.enter_bankruptcy() and self.market_exit()
           of this class (metainsuranceorg.py). First of all it dissolves all the contracts currently held (those in
           self.underwritten_contracts).
           Next all the cash currently available is transferred to insurancesimulation.py through an obligation in the
           next iteration. Finally the type of dissolution is recorded and the operational state is set to false.
           Different class variables are reset during the process: self.risks_kept, self.reinrisks_kept, self.excess_capital
           and self.profits_losses."""
        # Record all unpaid claims (needs to be here to account for firms lost due to regulator/being sold)
        if record != "record_market_exit":      # Market exits already pay all obligations
            sum_due = sum(item.amount for item in self.obligations)  # Also records dividends/premiums
            self.simulation.record_unrecovered_claims(sum_due - self.cash)

        # Removing (dissolving) all risks immediately after bankruptcy (may not be realistic, they might instead be bought by another company)
        [contract.dissolve(time) for contract in self.underwritten_contracts]
        self.simulation.return_risks(self.risks_kept)
        self.risks_kept = []
        self.reinrisks_kept = []

        obligation = Obligation(amount=self.cash, recipient=self.simulation, due_time=time, purpose="Dissolution")
        self._pay(obligation)  # This MUST be the last obligation before the dissolution of the firm.

        self.excess_capital = 0  # Excess of capital is 0 after bankruptcy or market exit.
        self.profits_losses = 0  # Profits and losses are 0 after bankruptcy or market exit.

        method_to_call = getattr(self.simulation, record)
        method_to_call()
        for reincontract in self.reinsurance_profile.all_contracts():
            reincontract.dissolve(time)
        self.operational = False

    def pay_dividends(self, time: int):
        """Method to receive dividend obligation.
                    Accepts:
                        time: Type integer
                    No return value
                    If firm has positive profits will pay percentage of them as dividends.
                    Currently pays to simulation.
                    """
        self.receive_obligation(self.per_period_dividend, self.simulation, due_time=time, purpose="dividend")

    def mature_contracts(self, time: int) -> int:
        """Method to mature underwritten contracts that have expired
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

    def get_cash(self) -> float:
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

    def number_underwritten_contracts(self) -> int:
        return len(self.underwritten_contracts)

    def get_underwritten_contracts(self) -> Sequence["MetaInsuranceContract"]:
        return self.underwritten_contracts

    def get_profitslosses(self) -> float:
        """Method to get agents profit or loss. Only used for saving data. Called by simulation.
            No Accepted values.
            Returns agents profits/losses"""
        return self.profits_losses

    def estimate_var(self) -> None:
        """Method to estimate Value at Risk.
            No Accepted arguments.
            No return values
           Calculates value at risk per category and overall, based on underwritten contracts initial value at risk.
           Assigns it to agent instance. Called at the end of each agents iteration cycle. Also records the VaR and
           reinsurance contract info for the last 12 iterations, used for regulation."""
        self.counter_category = np.zeros(self.simulation_no_risk_categories)
        self.var_category = np.zeros(self.simulation_no_risk_categories)

        self.var_counter = 0
        self.var_counter_per_risk = 0
        self.var_sum = 0
        current_reinsurance_info = []

        # Extract initial VaR per category
        for contract in self.underwritten_contracts:
            self.counter_category[contract.category] += 1
            self.var_category[contract.category] += contract.initial_VaR

        # Calculate risks per category and sum of all VaR
        for category in range(len(self.counter_category)):
            self.var_counter += self.counter_category[category] * self.riskmodel.inaccuracy[category]
            self.var_sum += self.var_category[category]

        # Record reinsurance info
        for region_list in self.reinsurance_profile.reinsured_regions:
            current_region_info = []
            if len(region_list) > 0:
                for contract in region_list:
                    current_region_info.append([contract[0], contract[1]])
            else:
                current_region_info.append([0,0])
            current_reinsurance_info.append(current_region_info)

        # Rotate lists and replace for up-to-date list for 12 iterations
        self.var_sum_last_periods = np.roll(self.var_sum_last_periods, -4)
        self.var_sum_last_periods[-1] = self.var_category
        self.reinsurance_history.append(current_reinsurance_info)
        self.reinsurance_history.pop(0)

        # Calculate average no. risks per category
        if sum(self.counter_category) != 0:
            self.var_counter_per_risk = self.var_counter / sum(self.counter_category)
        else:
            self.var_counter_per_risk = 0

    def get_newrisks_by_type(self) -> Tuple[Sequence[RiskProperties], Sequence[RiskProperties]]:
        """Method for soliciting new risks from insurance simulation then organising them based if non-proportional
            or not.
            No accepted Values.
            Returns:
                new_non_proportional_risks: Type list of DataDicts.
                new_risks: Type list of DataDicts."""
        new_risks = []
        if self.is_insurer:
            new_risks += self.simulation.solicit_insurance_requests(self)
        if self.is_reinsurer:
            new_risks += self.simulation.solicit_reinsurance_requests(self)

        new_nonproportional_risks = [
            risk
            for risk in new_risks
            if risk.insurancetype == "excess-of-loss" and risk.owner is not self
        ]
        new_risks = [
            risk
            for risk in new_risks
            if risk.insurancetype in ["proportional", None] and risk.owner is not self
        ]
        return new_nonproportional_risks, new_risks

    def update_risk_characterisations(self):
        for categ in range(self.simulation_no_risk_categories):
            self.underwritten_risk_characterisation[categ] = self.characterise_underwritten_risks_by_category(categ)

    def characterise_underwritten_risks_by_category(self, categ_id: int) -> Tuple[float, float, int, float]:
        """Method to characterise associated risks in a given category in terms of value, number, avg risk factor, and
        total premium per iteration.
            Accepts:
                categ_id: Type Integer. The given category for characterising risks.
            Returns:
                total_value: Type Decimal. Total value of all contracts in the category.
                avg_risk_facotr: Type Decimal. Avg risk factor of all contracted risks in category.
                number_risks: Type Integer. Total number of contracted risks in category.
                periodised_total_premium: Total value per month of all contracts premium payments."""
        # TODO: Update this instead of recalculating so much
        total_value = 0
        avg_risk_factor = 0
        number_risks = 0
        periodized_total_premium = 0
        for contract in self.underwritten_contracts:
            if contract.category == categ_id:
                total_value += contract.value
                avg_risk_factor += contract.risk_factor
                number_risks += 1
                periodized_total_premium += contract.periodized_premium
        if number_risks > 0:
            avg_risk_factor /= number_risks
        return total_value, avg_risk_factor, number_risks, periodized_total_premium

    def risks_reinrisks_organizer(self, new_risks: Sequence[RiskProperties]) -> Sequence[Sequence[RiskProperties]]:
        """This method organizes the new risks received by the insurer (or reinsurer) by category.
                    Accepts:
                        new_risks: Type list of DataDicts
                    Returns:
                        risks_by_category: Type list of categories, each contains risks originating from that category.
                        number_risks_categ: Type list, elements are integers of total risks in each category"""
        risks_by_category = [
            [] for _ in range(self.simulation_parameters["no_categories"])
        ]
        number_risks_categ = np.zeros(
            self.simulation_parameters["no_categories"], dtype=np.int_
        )

        for categ_id in range(self.simulation_parameters["no_categories"]):
            risks_by_category[categ_id] = [
                risk for risk in new_risks if risk.category == categ_id
            ]
            number_risks_categ[categ_id] = len(risks_by_category[categ_id])

        # The method returns both risks_by_category and number_risks_categ.
        return risks_by_category

    def balanced_portfolio(self, risk: RiskProperties, cash_left_by_categ: np.ndarray, var_per_risk: Optional[Sequence[float]]) -> Tuple[bool, np.ndarray]:
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

        if risk.insurancetype == "excess-of-loss":
            percentage_value_at_risk = self.riskmodel.get_ppf(
                categ_id=risk.category, tail_size=self.riskmodel.var_tail_prob
            )
            var_damage = (
                percentage_value_at_risk
                * risk.value
                * risk.risk_factor
                * self.riskmodel.inaccuracy[risk.category]
            )
            var_claim = (
                min(var_damage, risk.value * risk.limit_fraction)
                - risk.value * risk.deductible_fraction
            )

            # record liquidity requirement and apply margin of safety for liquidity requirement

            # Compute how the cash reserved by category would change if the new reinsurance risk was accepted
            cash_reserved_by_categ_store[risk.category] += (
                var_claim * self.riskmodel.margin_of_safety
            )

        else:
            # Compute how the cash reserved by category would change if the new insurance risk was accepted
            cash_reserved_by_categ_store[risk.category] += var_per_risk[risk.category]

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

    def process_newrisks_reinsurer(self, reinrisks_per_categ: Sequence[Sequence[RiskProperties]], time: int):
        """Method to decide if new risks are underwritten for the reinsurance firm.
            Accepts:
                reinrisks_per_categ: Type List of lists containing new reinsurance risks.
                time: Type integer
            No return values.
           This method processes one by one the reinrisks contained in reinrisks_per_categ in order to decide whether
           they should be underwritten or not. It is done in this way to maintain the portfolio as balanced as possible.
           For that reason we process risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... and so forth. If
           risks are accepted then a contract is written."""
        not_accepted_reinrisks = [[] for _ in range(len(reinrisks_per_categ))]
        has_accepted_risks = False
        for risk in roundrobin(reinrisks_per_categ):
            # Here we take only one risk per category at a time to achieve risk[C1], risk[C2], risk[C3],
            # risk[C4], risk[C1], risk[C2], ... if possible.
            assert risk
            # TODO: Move this out of the loop (maybe somewhere else entirely) and update it when needed
            underwritten_risks = [
                RiskProperties(
                    owner=self,
                    value=contract.value,
                    category=contract.category,
                    risk_factor=contract.risk_factor,
                    deductible=contract.deductible,
                    limit=contract.limit,
                    insurancetype=contract.insurancetype,
                    runtime_left=(contract.expiration - time),
                )
                for contract in self.underwritten_contracts
                if contract.insurancetype == "excess-of-loss"
            ]
            accept, cash_left_by_categ, var_this_risk, self.excess_capital = self.riskmodel.evaluate(
                underwritten_risks, self.cash, risk)
            # TODO: change riskmodel.evaluate() to accept new risk to be evaluated and
            #  to account for existing non-proportional risks correctly -> DONE.
            if accept:
                # TODO: rename this to per_value_premium in insurancecontract.py to avoid confusion
                per_value_reinsurance_premium = (
                    self.np_reinsurance_premium_share * risk.periodized_total_premium * risk.runtime
                    * (self.simulation.get_market_reinpremium() / self.simulation.get_market_premium()) / risk.value)
                # Here it is check whether the portfolio is balanced or not if the reinrisk
                # (risk_to_insure) is underwritten. Return True if it is balanced. False otherwise.
                condition, cash_left_by_categ = self.balanced_portfolio(risk, cash_left_by_categ, None)
                if condition:
                    contract = reinsurancecontract.ReinsuranceContract(self, risk, time, per_value_reinsurance_premium,
                                                                       risk.runtime, self.default_contract_payment_period,
                                                                       expire_immediately=self.simulation_parameters[
                                                                       "expire_immediately"], initial_var=var_this_risk,
                                                                       insurancetype=risk.insurancetype,)
                    # TODO: implement excess of loss for reinsurance contracts
                    self.underwritten_contracts.append(contract)
                    has_accepted_risks = True
                    self.cash_left_by_categ = cash_left_by_categ
                else:
                    not_accepted_reinrisks[risk.category].append(risk)
            else:
                not_accepted_reinrisks[risk.category].append(risk)

        return has_accepted_risks, not_accepted_reinrisks

    def process_newrisks_insurer(self, risks_per_categ: Sequence[Sequence[RiskProperties]],
                                 acceptable_by_category: Sequence[int],var_per_risk_per_categ: Sequence[float],
                                 cash_left_by_categ: Sequence[float],time: int,) -> Tuple[bool, Sequence[Sequence[RiskProperties]]]:
        """Method to decide if new risks are underwritten for the insurance firm.
            Accepts:
                risks_per_categ: Type List of lists containing new risks.
                acceptable_per_category:
                var_per_risk_per_categ: Type list of integers contains VaR for each category defined in getPPF.
                cash_left_by_categ:  Type List, contains list of available cash per category
                time: Type integer.
            Returns:
                risks_per_categ: Type list of list, same as above however with None where contracts were accepted.
                not_accepted_risks: Type List of DataDicts
        This method processes one by one the risks contained in risks_per_categ in order to decide whether
        they should be underwritten or not. It is done in this way to maintain the portfolio as balanced as possible.
        For that reason we process risk[C1], risk[C2], risk[C3], risk[C4], risk[C1], risk[C2], ... and so forth. If
        risks are accepted then a contract is written."""
        random_runtime = self.contract_runtime_dist.rvs()
        not_accepted_risks = [[] for _ in range(len(risks_per_categ))]
        has_accepted_risks = False
        for risk in roundrobin(risks_per_categ):
            assert risk
            if acceptable_by_category[risk.category] > 0:
                if risk.contract and risk.contract.expiration > time:
                    # In this case the risk being inspected already has a contract, so we are deciding whether to
                    # give reinsurance for it # QUERY: is this correct?
                    [condition, cash_left_by_categ] = self.balanced_portfolio(risk, cash_left_by_categ, None)
                    # Here we check whether the portfolio is balanced or not if the reinrisk (risk_to_insure) is
                    # underwritten. Return True if it is balanced. False otherwise.
                    if condition:
                        contract = reinsurancecontract.ReinsuranceContract(
                            self,
                            risk,
                            time,
                            self.insurance_premium(),
                            risk.expiration - time,
                            self.default_contract_payment_period,
                            expire_immediately=self.simulation_parameters["expire_immediately"])
                        self.underwritten_contracts.append(contract)
                        has_accepted_risks = True
                        self.cash_left_by_categ = cash_left_by_categ
                        acceptable_by_category[risk.category] -= 1
                    else:
                        not_accepted_risks[risk.category].append(risk)

                else:
                    [condition, cash_left_by_categ] = self.balanced_portfolio(risk, cash_left_by_categ, var_per_risk_per_categ)
                    # In this case there is no contact currently associated with the risk, so we decide whether
                    # to insure it
                    # Here it is check whether the portfolio is balanced or not if the risk (risk_to_insure) is
                    # underwritten. Return True if it is balanced. False otherwise.
                    if condition:
                        contract = insurancecontract.InsuranceContract(self, risk, time, self.simulation.get_market_premium(),
                                                    random_runtime, self.default_contract_payment_period,
                                                    expire_immediately=self.simulation_parameters["expire_immediately"],
                                                    initial_var=var_per_risk_per_categ[risk.category])
                        self.underwritten_contracts.append(contract)
                        has_accepted_risks = True
                        self.cash_left_by_categ = cash_left_by_categ
                        acceptable_by_category[risk.category] -= 1
                    else:
                        not_accepted_risks[risk.category].append(risk)
            else:
                not_accepted_risks[risk.category].append(risk)
                # TODO: allow different values per risk (i.e. sum over value (and reinsurance_share) or
                #  exposure instead of counting)
            # QUERY: should we only decrease this if the risk is accepted?
        return has_accepted_risks, not_accepted_risks

    def market_permanency(self, time: int):
        """Method determining if firm stays in market.
            Accepts:
                Time: Type Integer
            No return values. This method determines whether an insurer or reinsurer stays in the market.
        If it has very few risks underwritten or too much cash left for TOO LONG it eventually leaves the market.
        If it has very few risks underwritten it cannot balance the portfolio so it makes sense to leave the market."""
        if not self.simulation_parameters["market_permanency_off"]:

            cash_left_by_categ = np.asarray(self.cash_left_by_categ)

            avg_cash_left = get_mean(cash_left_by_categ)

            if self.cash < self.simulation_parameters["cash_permanency_limit"]:  # If their level of cash is so low that they cannot underwrite anything they also leave the market.
                self.market_exit(time)
            else:
                if self.is_insurer:
                    if len(self.underwritten_contracts) < self.simulation_parameters["insurance_permanency_contracts_limit"] or avg_cash_left / self.cash > self.simulation_parameters["insurance_permanency_ratio_limit"]:
                        # Insurers leave the market if they have contracts under the limit or an excess capital
                        # over the limit for too long.
                        self.market_permanency_counter += 1
                    else:
                        self.market_permanency_counter = 0
                    if self.market_permanency_counter >= self.simulation_parameters["insurance_permanency_time_constraint"]:
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

                    if self.market_permanency_counter >= self.simulation_parameters["reinsurance_permanency_time_constraint"]:  # Here we determine how much is too long.
                        self.market_exit(time)

    def register_claim(self, claim: float):
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

    def roll_over(self, time: int):
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
                    if next(uniform_rvs) > self.simulation_parameters["insurance_retention"]:
                        self.simulation.return_risks([contract.risk])  # TODO: This is not a retention, so the roll_over_flag might be confusing in this case
                    else:
                        self.risks_kept.append(contract.risk)

            if self.is_reinsurer:
                for reincontract in maturing_next:
                    if reincontract.property_holder.operational:
                        reincontract.roll_over_flag = 1
                        reinrisk = reincontract.property_holder.refresh_reinrisk(time=time, old_contract=reincontract)
                        if next(uniform_rvs)< self.simulation_parameters["reinsurance_retention"]:
                            if reinrisk is not None:
                                self.reinrisks_kept.append(reinrisk)

    def update_risk_share(self):
        """Updates own value for share of all risks held by this firm. Has neither arguments nor a return value"""
        self.risk_share = self.simulation.get_risk_share(self)

    def insurance_premium(self) -> float:
        """Returns the premium this firm will charge for insurance.

        Returns the market premium multiplied by a factor that scales linearly with self.risk_share between 1 and
        the max permissble adjustment"""
        max_adjustment = isleconfig.simulation_parameters["max_scale_premiums"]
        premium = self.simulation.get_market_premium() * (1 * (1 - self.risk_share) + max_adjustment * self.risk_share)
        return premium

    def adjust_riskmodel_inaccuracy(self):
        """Adjusts the inaccuracy parameter in the risk model in use depending on the share of risks held.
        Accepts no parameters and has no return

        Shrinks the risk model towards the best available risk model (as determined by "scale_inaccuracy" in isleconfig)
        by the share of risk this firm holds.
        """
        if isleconfig.simulation_parameters["scale_inaccuracy"] != 1:
            self.riskmodel.inaccuracy = (self.max_inaccuracy * (1 - self.risk_share) + self.min_inaccuracy * self.risk_share)

    def consider_buyout(self, type="insurer"):
        """Method to allow firm to decide if to buy one of the firms going bankrupt.
            Accepts:
                type: Type string. Used to decide if insurance or reinsurance firm.
            No return values.
        This method is called for both types of firms to consider buying one firm going bankrupt for only this iteration
        It has a chance (based on market share) to buyout other firm if its excess capital is large enough to cover
        the other firms value at risk multiplied by its margin of safety. Will call buyout() if necessary."""
        firms_to_consider = self.simulation.get_firms_to_sell(type)
        firms_further_considered = []

        for firm, time, reason in firms_to_consider:
            all_firms_cash = self.simulation.get_total_firm_cash(type)
            all_obligations = sum([obligation["amount"] for obligation in firm.obligations])
            total_premium = sum([np.mean(contract.payment_values) for contract in firm.underwritten_contracts if len(contract.payment_values) > 0])
            if self.excess_capital > self.riskmodel.margin_of_safety * firm.var_sum + all_obligations - total_premium:
                firm_likelihood = 0.25 + (1.5 * firm.cash + np.mean(firm.cash_last_periods[1:12]) + self.cash)/all_firms_cash
                firm_likelihood = min(1, 2*firm_likelihood)
                firm_price = (firm.var_sum/10) + total_premium + firm.per_period_dividend
                firm_sell_reason = reason
                firms_further_considered.append([firm, firm_likelihood, firm_price, firm_sell_reason])

        if len(firms_further_considered) > 0:
            best_likelihood = 0
            for firm_data in firms_further_considered:
                if firm_data[1] > best_likelihood:
                    best_likelihood = firm_data[1]
                    best_firm = firm_data[0]
                    best_firm_cost = firm_data[2]
                    best_firm_sell_reason = firm_data[3]
            random_chance = np.random.uniform(0, 1)
            if best_likelihood > random_chance:
                self.buyout(best_firm, best_firm_cost, time)
                self.simulation.remove_sold_firm(best_firm, time, best_firm_sell_reason)

    def buyout(self, firm, firm_cost, time):
        """Method called to actually buyout firm.
        Accepts:
            firm: Type Class. Firm being bought.
            firm_cost: Type Decimal. Cost of firm being bought.
            time: Type Integer. Time at which bought.
        No return values.
    This method causes buyer to receive obligation to buy firm. Sets all the bought firms contracts as its own. Then
    clears bought firms contracts and dissolves it. Only called from consider_buyout()."""
        self.receive_obligation(firm_cost, self.simulation, time, 'buyout')

        if self.is_insurer and firm.is_insurer:
            print("Insurer %i has bought %i for %d with total cash %d" % (self.id, firm.id, firm_cost, self.cash))
        elif self.is_reinsurer and firm.is_reinsurer:
            print("Reinsurer %i has bought %i  for %d with total cash %d" % (self.id, firm.id, firm_cost, self.cash))

        for contract in firm.underwritten_contracts:
            contract.insurer = self
            self.underwritten_contracts.append(contract)
        for obli in firm.obligations:
            self.receive_obligation(obli['amount'], obli["recipient"], obli["due_time"], obli["purpose"])

        firm.obligations = []
        firm.underwritten_contracts = []
        firm.dissolve(time, 'record_bought_firm')

    def submit_regulator_report(self, time):
        """Method to submit cash, VaR, and reinsurance data to central banks regulate(). Sets a warning or triggers
        selling of firm if not complying with regulation (holding enough effective capital for risk).
            No accepted values.
            No return values."""
        condition = self.simulation.bank.regulate(self.id, self.cash_last_periods, self.var_sum_last_periods,
                                                  self.reinsurance_history, self.age, self.riskmodel.margin_of_safety)
        if condition == "Good":
            self.warning = False
        if condition == "Warning":
            self.warning = True
        if condition == "LoseControl":
            if isleconfig.buy_bankruptcies:
                self.simulation.add_firm_to_be_sold(self, time, "record_nonregulation_firm")
                self.operational = False
            else:
                self.dissolve(time, "record_nonregulation_firm")
                for contract in self.underwritten_contracts:
                    contract.mature(time)
                self.underwritten_contracts = []

