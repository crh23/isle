import numpy as np

import metainsuranceorg
import catbond
from reinsurancecontract import ReinsuranceContract
import isleconfig
import genericclasses
from typing import Optional, Collection, Mapping

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    pass


class InsuranceFirm(metainsuranceorg.MetaInsuranceOrg):
    """ReinsuranceFirm class.
       Inherits from MetaInsuranceFirm."""

    def __init__(self, simulation_parameters, agent_parameters):
        """Constructor method.
               Accepts arguments
                   Signature is identical to constructor method of parent class.
           Constructor calls parent constructor and only overwrites boolean indicators of insurer and reinsurer role of
           the object."""
        super().__init__(simulation_parameters, agent_parameters)
        self.is_insurer = True
        self.is_reinsurer = False

    def adjust_dividends(self, time: int, actual_capacity: float):
        """Method to adjust dividends firm pays to investors.
            Accepts:
                time: Type Integer. Not used.
                actual_capacity: Type Decimal.
            No return values.
        Method is called from MetaInsuranceOrg iterate method between evaluating reinsurance and insurance risks to
        calculate dividend to be payed if the firm has made profit and has achieved capital targets."""

        profits = self.get_profitslosses()
        self.per_period_dividend = max(0, self.dividend_share_of_profits * profits)
        # max function ensures that no negative dividends are paid
        if actual_capacity < self.capacity_target:
            # no dividends if firm misses capital target
            self.per_period_dividend = 0

    def get_reinsurance_var_estimate(self, max_var: float) -> float:
        """Method to estimate the VaR if another reinsurance contract were to be taken out.
            Accepts:
                max_var: Type Decimal. Max value at risk
            Returns:
                reinsurance_VaR_estimate: Type Decimal.
        This method takes the max VaR and mulitiplies it by a factor that estimates the VaR if another reinsurance
        contract was to be taken. Called by the adjust_target_capacity and get_capacity methods."""
        # TODO: Should be total_value, or maybe the total amount of exposure (rather than number_risks)
        values = [
            self.underwritten_risk_characterisation[categ].number_risks
            for categ in range(self.simulation_parameters["no_categories"])
        ]
        reinsurance_factor_estimate = self.get_reinsurable_fraction(values)
        reinsurance_var_estimate = max_var * (1.0 + reinsurance_factor_estimate)
        return reinsurance_var_estimate

    def get_reinsurable_fraction(self, value_by_category):
        """Returns the proportion of the value of risk held overall that is eligible for reinsurance"""
        total = 0
        for categ in range(len(value_by_category)):
            value: float = value_by_category[categ]
            uncovered = self.reinsurance_profile.uncovered(categ)
            maximum_excess: float = self.np_reinsurance_limit_fraction * value
            miniumum_deductible: float = self.np_reinsurance_deductible_fraction * value
            for region in uncovered:
                if region[1] > miniumum_deductible and region[0] < maximum_excess:
                    total += min(
                        region[1] / value, self.np_reinsurance_limit_fraction
                    ) - max(region[0] / value, self.np_reinsurance_deductible_fraction)
        total = total / len(value_by_category)
        return total

    def adjust_capacity_target(self, max_var: float):
        """Method to adjust capacity target.
                   Accepts:
                       max_var: Type Decimal.
                   No return values.
               This method decides to increase/decrease the capacity target dependant on if the ratio of
                capacity target to max VaR is above/below a predetermined limit."""
        reinsurance_var_estimate = self.get_reinsurance_var_estimate(max_var)
        if max_var + reinsurance_var_estimate == 0:
            # TODO: why is this being called with max_var = 0 anyway?
            capacity_target_var_ratio_estimate = np.inf
        else:
            capacity_target_var_ratio_estimate = (
                (self.capacity_target + reinsurance_var_estimate)
                * 1.0
                / (max_var + reinsurance_var_estimate)
            )
        if (
            capacity_target_var_ratio_estimate
            > self.capacity_target_increment_threshold
        ):
            self.capacity_target *= self.capacity_target_increment_factor
        elif (
            capacity_target_var_ratio_estimate
            < self.capacity_target_decrement_threshold
        ):
            self.capacity_target *= self.capacity_target_decrement_factor

    def get_capacity(self, max_var: float) -> float:
        """Method to get capacity of firm.
                    Accepts:
                        max_var: Type Decimal.
                    Returns:
                        self.cash (+ reinsurance_VaR_estimate): Type Decimal.
                This method is called by increase_capacity to get the real capacity of the firm. If the firm has
                enough money to cover its max value at risk then its capacity is its cash + the reinsurance VaR
                estimate, otherwise the firm is recovering from some losses and so capacity is just cash."""
        if max_var < self.cash:
            reinsurance_var_estimate = self.get_reinsurance_var_estimate(max_var)
            return self.cash + reinsurance_var_estimate
        else:
            # (This point is only reached when insurer is in severe financial difficulty.
            # Ensure insurer recovers complete coverage.)
            return self.cash

    def increase_capacity(self, time: int, max_var: float) -> float:
        """Method to increase the capacity of the firm.
            Accepts:
                time: Type Integer.
                max_var: Type Decimal.
            Returns:
                capacity: Type Decimal.
        This method is called from the main iterate method in metainsuranceorg and gets prices for cat bonds and
        reinsurance then checks if each category needs it. Passes a random category and the prices to the
        increase_capacity_by_category method. If a firms capacity is above its target then it will only issue one if the
        market premium is above its average premium, otherwise firm is 'forced' to get a catbond or reinsurance. Only
        implemented for non-proportional(excess of loss) reinsurance. Only issues one reinsurance or catbond per
        iteration unless not enough capacity to meet target."""
        assert self.simulation_reinsurance_type == "non-proportional"
        """get prices"""

        capacity = None
        if not (self.simulation.catbonds_off and self.simulation.reinsurance_off):
            categ_ids = list(range(self.simulation_no_risk_categories))
            np.random.shuffle(categ_ids)
            while len(categ_ids) >= 1:
                categ_id = categ_ids.pop()
                capacity = self.get_capacity(max_var)
                if self.capacity_target < capacity:
                    # just one per iteration, unless capital target is unmatched
                    if self.increase_capacity_by_category(time, categ_id, force=False):
                        categ_ids = []
                else:
                    self.increase_capacity_by_category(time, categ_id, force=True)
        # capacity is returned in order not to recompute more often than necessary
        if capacity is None:
            capacity = self.get_capacity(max_var)
        return capacity

    def increase_capacity_by_category(
        self, time: int, categ_id: int, force: bool = False
    ) -> bool:
        """Method to increase capacity. Only called by increase_capacity.
            Accepts:
                time: Type Integer
                categ_id: Type integer.
                force: Type Boolean. Forces firm to get reinsurance/catbond or not.
            Returns Boolean to stop loop if firm has enough capacity.
        This method is given a category and prices of reinsurance/catbonds and will issue whichever one is cheaper to a
        firm for the given category. This is forced if firm does not have enough capacity to meet target otherwise will
        only issue if market premium is greater than firms average premium."""
        if isleconfig.verbose:
            print(f"IF {self.id:d} increasing capacity in period {time:d}.")
        if not force:
            actual_premium = self.get_average_premium(categ_id)
            possible_premium = self.simulation.get_market_premium()
            if actual_premium >= possible_premium:
                return False
        """on the basis of prices decide for obtaining reinsurance or for issuing cat bond"""
        self.ask_reinsurance_non_proportional_by_category(time, categ_id)
        return True

    def get_average_premium(self, categ_id: int) -> float:
        """Method to calculate and return the firms average premium for all currently underwritten contracts.
            Accepts:
                categ_id: Type Integer.
            Returns:
                premium payments left/total value of contracts: Type Decimal"""
        # weighted_premium_sum = 0
        # total_weight = 0
        # for contract in self.underwritten_contracts:
        #     if contract.category == categ_id:
        #         total_weight += contract.value
        #         contract_premium = contract.periodized_premium * contract.runtime
        #         weighted_premium_sum += contract_premium

        total_weight = self.underwritten_risk_characterisation[
            categ_id
        ].total_value  # TODO: Should use exposure
        weighted_premium_sum = self.underwritten_risk_characterisation[
            categ_id
        ].weighted_premium
        if total_weight == 0:
            return 0  # will prevent any attempt to reinsure empty categories
        return weighted_premium_sum * 1.0 / total_weight

    def ask_reinsurance_non_proportional_by_category(
        self,
        time: int,
        categ_id: int,
        purpose: str = "newrisk",
        min_tranches: int = None,
    ) -> Optional[List[genericclasses.RiskProperties]]:
        """Method to create a reinsurance risk for a given category for firm that calls it. Called from increase_
        capacity_by_category, ask_reinsurance_non_proportional, and roll_over in metainsuranceorg.
            Accepts:
                time: Type Integer.
                categ_id: Type Integer.
                purpose: Type String. Needed for when called from roll_over method as the risk is then returned.
                min_tranches: Type int. Determines how many layers of reinsurance the risk is split over
            Returns:
                risk: Type DataDict. Only returned when method used for roll_over.
        This method is given a category, then characterises all the underwritten risks in that category for the firm
        and, assuming firms has underwritten risks in category, creates new reinsurance risks with values based on firms
        existing underwritten risks. If tranches > 1, the risk is split between mutliple layers of reinsurance, each of
         the same size. If the method was called to create a new risks then it is appended to list of
        'reinrisks', otherwise used for creating the risk when a reinsurance contract rolls over."""
        # TODO: how do we decide how many tranches?
        if min_tranches is None:
            min_tranches = isleconfig.simulation_parameters["min_tranches"]

        total_value = self.underwritten_risk_characterisation[categ_id].total_value
        number_risks = self.underwritten_risk_characterisation[categ_id].number_risks

        if number_risks > 0:
            tranches = self.reinsurance_profile.uncovered(categ_id)

            # Don't get reinsurance above maximum limit
            while tranches[-1][1] > self.np_reinsurance_limit_fraction * total_value:
                if tranches[-1][0] >= self.np_reinsurance_limit_fraction * total_value:
                    tranches.pop()
                else:
                    tranches[-1] = (
                        tranches[-1][0],
                        self.np_reinsurance_limit_fraction * total_value,
                    )
            while (
                tranches[0][0] < self.np_reinsurance_deductible_fraction * total_value
            ):
                if (
                    tranches[0][1]
                    <= self.np_reinsurance_deductible_fraction * total_value
                ):
                    tranches = tranches[1:]
                    if len(tranches) == 0:
                        break
                else:
                    tranches[0] = (
                        self.np_reinsurance_deductible_fraction * total_value,
                        tranches[0][1],
                    )
            for tranche in tranches[:]:
                # Use the slice so we aren't modifying while iterating
                if (tranche[1] - tranche[0]) <= max(
                    100,
                    0.1
                    * (
                        self.np_reinsurance_limit_fraction
                        - self.np_reinsurance_deductible_fraction
                    )
                    * total_value,
                ):
                    # Small gaps are acceptable to avoid having trivial contracts - we don't accept tranches with
                    # size less than 100 or 10% of the total reinsurable ammount
                    # TODO: the 10% limit should be removed if we have very many layers of reinsurance
                    tranches.remove(tranche)

            if not tranches:
                # If we've ended up with no tranches, give up and return
                return None

            while (
                len(tranches) + len(self.reinsurance_profile.all_contracts())
                < min_tranches
            ):
                # Make sure that the overall number of tranches after obtaining the requested reinsurance would be at
                # least the minimal value.
                tranches = self.reinsurance_profile.split_longest(tranches)
            risks_to_return = []
            for tranche in tranches:
                assert tranche[1] > tranche[0]
                risk = self.reinsure_tranche(
                    categ_id,
                    tranche[0] / total_value,
                    tranche[1] / total_value,
                    time,
                    purpose,
                )
                if purpose == "rollover":
                    risks_to_return.append(risk)
            if purpose == "rollover":
                return risks_to_return
        elif purpose == "rollover":
            return None

    def reinsure_tranche(
        self,
        category: int,
        deductible_fraction: float,
        limit_fraction: float,
        time: int,
        purpose: str,
    ):
        [
            total_value,
            avg_risk_factor,
            number_risks,
            periodized_total_premium,
            _,
        ] = self.underwritten_risk_characterisation[category]
        risk = genericclasses.RiskProperties(
            value=total_value,
            category=category,
            owner=self,
            insurancetype="excess-of-loss",
            number_risks=number_risks,
            deductible_fraction=deductible_fraction,
            limit_fraction=limit_fraction,
            periodized_total_premium=periodized_total_premium,
            runtime=12,
            expiration=time + 12,
            risk_factor=avg_risk_factor,
            deductible=deductible_fraction * total_value,
            limit=limit_fraction * total_value,
        )  # TODO: make runtime into a parameter
        reinsurance_type = self.decide_reinsurance_type(risk)
        if reinsurance_type == "reinsurance":
            if purpose == "newrisk":
                self.simulation.append_reinrisks(risk)
                return None
            elif purpose == "rollover":
                return risk

        elif reinsurance_type == "catbond":
            # The whole premium is transfered to the bond at creation, not periodically
            # TODO: Should the premium be periodic as for any other reinsurance? Would help, probably
            risk.periodized_total_premium = 0
            total_premium = (
                self.get_catbond_price(risk)
                * risk.value
                * self.np_reinsurance_premium_share
            )
            if not self.cash >= total_premium:
                # We can't actually afford to issue the catbond. Ideally this shouldn't be reached, but it is.
                return None
            per_period_premium = total_premium / risk.runtime
            new_catbond = catbond.CatBond(self.simulation, per_period_premium, self)

            """add contract; contract is a quasi-reinsurance contract"""
            # This automatically adds reinsurance to self.reinsurance_profile
            # per_value_reinsurance_premium = 0 because the insurance firm make only one payment to catbond
            contract = ReinsuranceContract(
                new_catbond,
                risk,
                time,
                0,
                risk.runtime,
                self.default_contract_payment_period,
                expire_immediately=self.simulation_parameters["expire_immediately"],
                insurancetype=risk.insurancetype,
            )

            new_catbond.set_contract(contract)

            """sell cat bond (to self.simulation)"""
            # amount changed from var_this_risk to total exposure
            exposure = risk.value * (risk.limit_fraction - risk.deductible_fraction)
            self.simulation.receive_obligation(exposure + 1, new_catbond, time, "bond")
            new_catbond.set_owner(self.simulation)

            """hand cash over to cat bond to cover the premium payouts"""
            # If we added this as an obligation in the normal way, there would be a risk that the firm would go under
            # before paying, which would cause the catbond to never really have been created which is confusing

            obligation = genericclasses.Obligation(
                amount=total_premium,
                recipient=new_catbond,
                due_time=time,
                purpose="bond",
            )
            self._pay(obligation)
            """register catbond"""
            self.simulation.add_agents(catbond.CatBond, "catbond", [new_catbond])
        else:
            # print(f"\nIF {self.id} attempted to get reinsurance for {risk.limit-risk.deductible:.0f} xs"
            #      f" {risk.deductible:.0f} but it was too expensive")
            return None

    def decide_reinsurance_type(self, risk: genericclasses.RiskProperties) -> str:
        """Decides whether to get catbond or reinsurance for risk with given properties"""
        # This should be the only place where VaR is evaluated. It should be moved out if we want to use it for
        # pricing etc.
        catbond_price = (
            self.get_catbond_price(risk)
            * risk.value
            * self.np_reinsurance_premium_share
        )
        reinsurance_price = (
            self.get_reinsurance_price(risk)
            * risk.value
            * self.np_reinsurance_premium_share
        )
        if catbond_price == reinsurance_price == float("inf"):
            return "nope"

        _, _, var_this_risk, _ = self.riskmodel.evaluate([], self.cash, risk)
        capacity_gain = var_this_risk * self.riskmodel.margin_of_safety
        if catbond_price < reinsurance_price:
            if capacity_gain < catbond_price:
                # If we won't actually gain any capacity due to the loss in capital, don't do it!
                # TODO: Does this make sense for reinsurance?
                return "nope"
            else:
                return "catbond"
        else:
            if capacity_gain < reinsurance_price:
                # TODO: This uses the total premium as the capacity loss due to premium expenditure - is this right?
                return "nope"
            else:
                return "reinsurance"

    def get_catbond_price(self, risk: genericclasses.RiskProperties) -> float:
        """Returns the total per-risk premium for a catbond """
        # TODO: take limit into account as well as deductible
        assert risk.deductible_fraction is not None
        return self.simulation.get_cat_bond_price(
            risk.deductible_fraction, risk.limit_fraction
        )

    def get_reinsurance_price(self, risk: genericclasses.RiskProperties) -> float:
        """Returns the total per-risk premium for reinsurance"""
        # TODO: take limit into account as well as deductible
        assert risk.deductible_fraction is not None
        return self.simulation.get_reinsurance_premium(
            risk.deductible_fraction, risk.limit_fraction
        )

    def add_reinsurance(self, contract: ReinsuranceContract, force_value: float = None):
        """Add reinsurance to the reinsurance profile. Value is given as contract.value is set when contract is offered,
        not when it is accepted.
        Value can be forced if we are updating an old contract rather than issuing a new one.
            Accepts:
                category: Type Integer.
                contract: Type Class. Reinsurance contract issued to firm.
            No return values."""
        if force_value is not None:
            value = force_value
        else:
            value = self.underwritten_risk_characterisation[
                contract.category
            ].total_value
        self.reinsurance_profile.add(contract, value)

    def delete_reinsurance(self, contract: ReinsuranceContract):
        """Method called by reinsurancecontract to delete the reinsurance contract from the firms counter for the given
        category, used so that another reinsurance contract can be issued for that category if needed.
            Accepts:
                category: Type Integer.
                contract: Type Class. Reinsurance contract issued to firm.
            No return values."""
        value = self.underwritten_risk_characterisation[contract.category].total_value
        self.reinsurance_profile.remove(contract, value)

    def make_reinsurance_claims(self, time: int):
        """Method to make reinsurance claims.
            Accepts:
                time: Type Integer.
            No return values.
        This method calculates the total amount of claims this iteration per category, and explodes (see reinsurance
        contracts) any reinsurance contracts present for one of the contracts (currently always zero). Then, for a
        category with reinsurance and claims, the applicable reinsurance contract is exploded."""
        # TODO: reorganize this with risk category ledgers
        # TODO: Put facultative insurance claims here
        claims_this_turn = np.zeros(self.simulation_no_risk_categories)
        for contract in self.underwritten_contracts:
            categ_id, claims, is_proportional = contract.get_and_reset_current_claim()
            if is_proportional:
                claims_this_turn[categ_id] += claims
            if contract.reincontract:
                contract.reincontract.explode(time, damage_extent=claims)

        for categ_id in range(self.simulation_no_risk_categories):
            if claims_this_turn[categ_id] > 0:
                to_explode = self.reinsurance_profile.contracts_to_explode(
                    damage=claims_this_turn[categ_id], category=categ_id
                )
                for contract in to_explode:
                    contract.explode(time, damage_extent=claims_this_turn[categ_id])

    def get_excess_of_loss_reinsurance(self) -> Collection[Mapping]:
        """Method to return list containing the reinsurance for each category interms of the reinsurer, value of
        contract and category. Only used for network visualisation.
            No accepted values.
            Returns:
                reinsurance: Type list of DataDicts."""
        reinsurance = []
        for contract in self.reinsurance_profile.all_contracts():
            reinsurance.append(
                {
                    "reinsurer": contract.insurer,
                    # QUERY: value vs excess?
                    "value": contract.value,
                    "category": contract.category,
                }
            )
        return reinsurance

    def refresh_reinrisk(
        self, time: int, old_contract: "ReinsuranceContract"
    ) -> Optional[genericclasses.RiskProperties]:
        # TODO: Can be merged
        """Takes an expiring contract and returns a renewed risk to automatically offer to the existing reinsurer.
        The new risk has the same deductible and excess as the old one, but with an updated time"""
        [
            total_value,
            avg_risk_factor,
            number_risks,
            periodized_total_premium,
            _,
        ] = self.underwritten_risk_characterisation[old_contract.category]
        if number_risks == 0:
            # If the insurerer currently has no risks in that category it probably doesn't want reinsurance
            return None
        risk = genericclasses.RiskProperties(
            value=total_value,
            category=old_contract.category,
            owner=self,
            insurancetype="excess-of-loss",
            number_risks=number_risks,
            deductible_fraction=old_contract.deductible / total_value,
            limit_fraction=old_contract.limit / total_value,
            periodized_total_premium=periodized_total_premium,
            runtime=12,
            expiration=time + 12,
            risk_factor=avg_risk_factor,
        )
        return risk


class ReinsuranceFirm(InsuranceFirm):
    """ReinsuranceFirm class.
       Inherits from InsuranceFirm."""

    def __init__(self, simulation_parameters, agent_parameters):
        """Constructor method.
               Accepts arguments
                   Signature is identical to constructor method of parent class.
           Constructor calls parent constructor and only overwrites boolean indicators of insurer and reinsurer role of
           the object."""
        super().__init__(simulation_parameters, agent_parameters)
        self.is_insurer = False
        self.is_reinsurer = True
