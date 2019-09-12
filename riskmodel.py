import math
import copy

import numpy as np

import isleconfig
from distributionreinsurance import ReinsuranceDistWrapper
from typing import Sequence, Tuple, Union, Optional, MutableSequence, Collection

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genericclasses import Distribution, RiskProperties


class RiskModel:
    def __init__(
        self,
        damage_distribution: "Distribution",
        expire_immediately: bool,
        cat_separation_distribution: "Distribution",
        norm_premium: float,
        category_number: int,
        init_average_exposure: float,
        init_average_risk_factor: float,
        init_profit_estimate: float,
        margin_of_safety: float,
        var_tail_prob: float,
        inaccuracy: Sequence[float],
    ) -> None:
        self.cat_separation_distribution = cat_separation_distribution
        self.norm_premium = norm_premium
        self.var_tail_prob = var_tail_prob
        self.expire_immediately = expire_immediately
        self.category_number = category_number
        self.init_average_exposure = init_average_exposure
        self.init_average_risk_factor = init_average_risk_factor
        self.init_profit_estimate = init_profit_estimate
        self.margin_of_safety = margin_of_safety
        """damage_distribution is some scipy frozen rv distribution which is bound between 0 and 1 and indicates
           the share of risks suffering damage as part of any single catastrophic peril"""
        self.damage_distribution: MutableSequence["Distribution"] = [
            damage_distribution for _ in range(self.category_number)
        ]
        self.underlying_distribution = copy.deepcopy(self.damage_distribution)
        self.inaccuracy: Sequence[float] = inaccuracy

    def get_ppf(self, categ_id: int, tail_size: float) -> float:
        """Method for getting quantile function of the damage distribution (value at risk) by category.
           Positional arguments:
              categ_id  integer:           category
              tailSize  (float 0<=x<=1):   quantile
           Returns value-at-risk."""
        return self.damage_distribution[categ_id].ppf(1 - tail_size)

    def get_risks_by_categ(
        self, risks: Sequence["RiskProperties"]
    ) -> Sequence[Sequence["RiskProperties"]]:
        """Method splits list of risks by category
                    Accepts:
                        risks: Type List of DataDicts
                    Returns:
                        categ_risks: Type List of DataDicts."""
        risks_by_categ = [[] for _ in range(self.category_number)]
        for risk in risks:
            risks_by_categ[risk.category].append(risk)
        return risks_by_categ

    def compute_expectation(
        self, categ_risks: Sequence["RiskProperties"], categ_id: int
    ) -> Tuple[float, float, float]:
        # TODO: more intuitive name?
        """Method to compute the average exposure and risk factor as well as the increase in expected profits for the
                risks in a given category.
                    Accepts:
                        categ_risks: Type List of DataDicts.
                        categ_id: Type Integer.
                    Returns:
                        average_risk_factor: Type Decimal.
                        average_exposure: Type Decimal. Mean risk factor in given category multiplied by inaccuracy.
                        incr_expected_profits: Type Decimal (currently only returns -1)"""
        exposures = np.zeros(len(categ_risks))
        risk_factors = np.zeros(len(categ_risks))
        runtimes = np.zeros(len(categ_risks))
        for i, risk in enumerate(categ_risks):
            # TODO: factor in excess instead of value?
            if risk.limit is None:
                raise ValueError("no no, no no no no, no no no no, there's no limit")
            exposures[i] = risk.value - risk.deductible
            risk_factors[i] = risk.risk_factor
            runtimes[i] = risk.runtime
        average_exposure: float = np.mean(exposures)
        average_risk_factor = self.inaccuracy[categ_id] * np.mean(risk_factors)

        # mean_runtime = np.mean(runtimes)

        if self.expire_immediately:
            incr_expected_profits = -1
            # TODO: fix the norm_premium estimation
            # incr_expected_profits = (
            #     (
            #         self.norm_premium
            #         - (
            #             1
            #             - scipy.stats.poisson(
            #                 1 / self.cat_separation_distribution.mean() * mean_runtime
            #             ).pmf(0)
            #         )
            #         * self.damage_distribution[categ_id].mean()
            #         * average_risk_factor
            #     )
            #     * average_exposure
            #     * len(categ_risks)
            # )
        else:
            incr_expected_profits = -1
            # TODO: expected profits should only be returned once the expire_immediately == False case is fixed
            # incr_expected_profits = (
            #     (
            #         self.norm_premium
            #         - mean_runtime
            #         / self.cat_separation_distribution[categ_id].mean()
            #         * self.damage_distribution.mean()
            #         * average_risk_factor
            #     )
            #     * average_exposure
            #     * len(categ_risks)
            # )

        return average_risk_factor, average_exposure, incr_expected_profits

    def evaluate_proportional(
        self, risks: Sequence["RiskProperties"], cash: Sequence[float]
    ) -> Tuple[float, Sequence[int], Sequence[int], Sequence[float]]:
        """Method to evaluate proportional type risks.
            Accepts:
                risks: Type List of DataDicts.
                cash: Type List. Gives cash available for each category.
            Returns:
                expected_profits: Type Decimal (Currently returns None)
                remaining_acceptable_by_category: Type List of Integers. Number of risks that would not be covered by
                    firms cash.
                cash_left_by_category: Type List of Integers. Firms expected cash left if underwriting the risks from
                    that category.
                var_per_risk_per_categ: List of Integers. Average VaR per category.
        This method iterates through the risks in each category and calculates the average VaR, how many could be
        underwritten according to their average VaR, how much cash would be left per category if all risks were
        underwritten at average VaR, and the total expected profit (currently always None)."""
        if not len(cash) == self.category_number:
            raise ValueError("Cash should be split category-wise")

        # prepare variables
        acceptable_by_category = []
        remaining_acceptable_by_category = []
        cash_left_by_category = np.copy(cash)
        expected_profits = 0
        necessary_liquidity = 0

        var_per_risk_per_categ = np.zeros(self.category_number)
        risks_by_categ = self.get_risks_by_categ(risks)
        # compute acceptable risks by category
        for categ_id in range(self.category_number):
            # compute number of acceptable risks of this category
            categ_risks = risks_by_categ[categ_id]
            if len(categ_risks) > 0:
                average_risk_factor, average_exposure, incr_expected_profits = self.compute_expectation(
                    categ_risks=categ_risks, categ_id=categ_id
                )
            else:
                average_risk_factor = self.init_average_risk_factor
                average_exposure = self.init_average_exposure
                incr_expected_profits = -1
                # TODO: expected profits should only be returned once the expire_immediately == False case is fixed

            expected_profits += incr_expected_profits

            if average_exposure == 0:
                average_exposure = self.init_average_exposure
            # compute value at risk
            var_per_risk = (
                self.get_ppf(categ_id=categ_id, tail_size=self.var_tail_prob)
                * average_risk_factor
                * average_exposure
                * self.margin_of_safety
            )

            # record liquidity requirement and apply margin of safety for liquidity requirement
            necessary_liquidity += var_per_risk * len(categ_risks)
            if isleconfig.verbose:
                print(self.inaccuracy)
                print(
                    "RISKMODEL: ",
                    var_per_risk,
                    " = PPF(0.02) * ",
                    average_risk_factor,
                    " * ",
                    average_exposure,
                    " vs. cash: ",
                    cash[categ_id],
                    "TOTAL_RISK_IN_CATEG: ",
                    var_per_risk * len(categ_risks),
                )
            acceptable = int(math.floor(cash[categ_id] / var_per_risk))
            remaining = acceptable - len(categ_risks)
            cash_left = cash[categ_id] - len(categ_risks) * var_per_risk

            acceptable_by_category.append(acceptable)
            remaining_acceptable_by_category.append(remaining)
            cash_left_by_category[categ_id] = cash_left
            var_per_risk_per_categ[categ_id] = var_per_risk

        # TODO: expected profits should only be returned once the expire_immediately == False case is fixed;
        #  the else-clause conditional statement should then be raised to unconditional
        if expected_profits < 0:
            expected_profits = None
        else:
            if necessary_liquidity == 0:
                if not expected_profits == 0:
                    raise ValueError(
                        "Expected profits should be zero at this point, but isn't"
                    )
                expected_profits = self.init_profit_estimate * cash[0]
            else:
                expected_profits /= necessary_liquidity

        if isleconfig.verbose:
            print(
                "RISKMODEL returns: ",
                expected_profits,
                remaining_acceptable_by_category,
            )
        return (
            expected_profits,
            remaining_acceptable_by_category,
            cash_left_by_category,
            var_per_risk_per_categ,
        )

    def evaluate_excess_of_loss(
        self,
        risks: Sequence["RiskProperties"],
        cash: Sequence[float],
        offered_risk: Optional["RiskProperties"] = None,
    ) -> Tuple[Sequence[float], Sequence[float], float]:
        """Method to evaluate excess-of-loss type risks.
                Accepts:
                    risks: Type List of DataDicts.
                    cash: Type List. Gives cash available for each category.
                    offered risk: Type DataDict
                Returns:
                    additional_required: Type List of Decimals. Capital required to cover offered risks potential claim
                                         (including margin of safety) per category. Only one will be non-zero.
                    cash_left_by_category: Type List of Decimals. Cash left per category if all risks claimed.
                    var_this_risk: Type Decimal. Expected claim of offered risk.
        This method iterates through the risks in each category and calculates the cash left in each category if
        each underwritten contract were to be claimed at expected values. The additional cash required to cover the
        offered risk (if applicable) is then calculated (should only be one)."""
        cash_left_by_categ = np.copy(cash)
        if not len(cash_left_by_categ) == self.category_number:
            raise ValueError("cash left not split by category")
        # prepare variables
        additional_required = np.zeros(self.category_number)
        additional_var_per_categ = np.zeros(self.category_number)

        risks_by_categ = self.get_risks_by_categ(risks)
        # values at risk and liquidity requirements by category
        for categ_id in range(self.category_number):
            categ_risks = risks_by_categ[categ_id]

            percentage_value_at_risk = self.get_ppf(
                categ_id=categ_id, tail_size=self.var_tail_prob
            )

            # compute liquidity requirements from existing contracts
            for risk in categ_risks:
                var_damage = (
                    percentage_value_at_risk
                    * risk.value
                    * risk.risk_factor
                    * self.inaccuracy[categ_id]
                )

                var_claim = max(min(var_damage, risk.limit) - risk.deductible, 0)

                # record liquidity requirement and apply margin of safety for liquidity requirement
                cash_left_by_categ[categ_id] -= var_claim * self.margin_of_safety

            # compute additional liquidity requirements from newly offered contract
            if (offered_risk is not None) and (offered_risk.category == categ_id):
                var_damage_fraction = (
                    percentage_value_at_risk
                    * offered_risk.risk_factor
                    * self.inaccuracy[categ_id]
                )
                var_claim_fraction = max(
                    min(var_damage_fraction, offered_risk.limit_fraction)
                    - offered_risk.deductible_fraction,
                    0,
                )
                var_claim_total = var_claim_fraction * offered_risk.value

                # record liquidity requirement and apply margin of safety for liquidity requirement
                additional_required[categ_id] += var_claim_total * self.margin_of_safety
                additional_var_per_categ[categ_id] += var_claim_total

        # Additional value at risk should only occur in one category. Assert that this is the case.
        if not sum(additional_var_per_categ > 0) <= 1:
            raise ValueError("Additional VaR in multiple categories")
        var_this_risk = max(additional_var_per_categ)

        return cash_left_by_categ, additional_required, var_this_risk

    # noinspection PyUnboundLocalVariable
    def evaluate(
        self,
        risks: Collection["RiskProperties"],
        cash: Union[float, Sequence[float]],
        offered_risk: Optional["RiskProperties"] = None,
    ) -> Union[
        Tuple[float, Sequence[int], Sequence[float], Sequence[float], float],
        Tuple[bool, Sequence[float], float, float],
    ]:
        """Method to evaluate given risks and the offered risk.
            Accepts:
                risks: List of DataDicts.
                cash: Type Decimal.
            Optional:
                offered_risk: Type DataDict or defaults to None.
            (offered_risk = None)
            Returns:
                expected_profits_proportional: Type Decimal (Currently returns None)
                remaining_acceptable_by_categ:  Type List of Integers. Number of risks that would not be covered by
                    firms cash.
                cash_left_by_categ: Type List of Integers. Firms expected cash left if underwriting the risks from that
                    category.
                var_per_risk_per_categ: List of Integers. Average VaR per category
                min(cash_left_by_categ): Type Decimal. Minimum
            (offered_risk != None) Returns:
                (cash_left_by_categ - additional_required > 0).all(): Type Boolean. Returns True only if all categories
                    have enough to cover the additional capital to insure risk.
                cash_left_by_categ: Type List of Decimals. Cash left per category if all risks claimed.
                var_this_risk: Type Decimal. Expected claim of offered risk.
                min(cash_left_by_categ): Type Decimal. Minimum value of cash left in a category after covering all
                    expected claims.
        This method organises all risks by insurance type then delegates then to respective methods
        (evaluate_prop/evaluate_excess_of_loss). Excess of loss risks are processed one at a time and are admitted using
        the offered_risk argument, whereas proportional risks are processed all at once leaving offered_risk = 0. This
        results in two sets of return values being used. These return values are what is used to determine if risks are
        underwritten or not."""
        # TODO: split this into two functions
        # ensure that any risk to be considered supplied directly as argument is non-proportional/excess-of-loss
        if not (
            (offered_risk is None) or offered_risk.insurancetype == "excess-of-loss"
        ):
            raise ValueError("proportional risk isn't evaluated like this")
        # construct cash_left_by_categ as a sequence, defining remaining liquidity by category
        if not isinstance(cash, (np.ndarray, list)):
            cash_left_by_categ = np.ones(self.category_number) * cash
        else:
            cash_left_by_categ = np.copy(cash)
        if not len(cash_left_by_categ) == self.category_number:
            raise ValueError("cash left by categ has wrong length")

        # sort current contracts
        el_risks = [risk for risk in risks if risk.insurancetype == "excess-of-loss"]
        risks = [risk for risk in risks if risk.insurancetype == "proportional"]
        # compute liquidity requirements and acceptable risks from existing contract
        if (offered_risk is not None) or (len(el_risks) > 0):
            cash_left_by_categ, additional_required, var_this_risk = self.evaluate_excess_of_loss(
                el_risks, cash_left_by_categ, offered_risk
            )
        if (offered_risk is None) or (len(risks) > 0):
            [
                expected_profits_proportional,
                remaining_acceptable_by_categ,
                cash_left_by_categ,
                var_per_risk_per_categ,
            ] = self.evaluate_proportional(risks, cash_left_by_categ)
        if offered_risk is None:
            # return numbers of remaining acceptable risks by category
            return (
                expected_profits_proportional,
                remaining_acceptable_by_categ,
                cash_left_by_categ,
                var_per_risk_per_categ,
                min(cash_left_by_categ),
            )
        else:
            # return boolean value whether the offered excess_of_loss risk can be accepted
            if isleconfig.verbose:
                print(
                    "REINSURANCE RISKMODEL",
                    cash,
                    cash_left_by_categ,
                    (cash_left_by_categ - additional_required > 0).all(),
                )
            # if not (cash_left_by_categ - additional_required > 0).all():
            #    pdb.set_trace()
            return (
                (cash_left_by_categ - additional_required > 0).all(),
                cash_left_by_categ,
                var_this_risk,
                min(cash_left_by_categ),
            )

    def set_reinsurance_coverage(
        self,
        value: float,
        coverage: MutableSequence[Tuple[float, float]],
        category: int,
    ):
        """Updates the riskmodel for the category given to have the reinsurance given by coverage"""
        # sometimes value==0, in which case we don't try to update the distribution
        # (as the current coverage is effectively infinite)
        if value > 0:
            self.damage_distribution[category] = ReinsuranceDistWrapper(
                self.underlying_distribution[category], coverage=coverage, value=value
            )
