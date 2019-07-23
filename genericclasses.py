from itertools import chain

import dataclasses
from sortedcontainers import SortedList
import numpy as np
from scipy import stats

import isleconfig

from typing import Mapping, MutableSequence, Union, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metainsurancecontract import MetaInsuranceContract
    from distributiontruncated import TruncatedDistWrapper
    from distributionreinsurance import ReinsuranceDistWrapper
    from reinsurancecontract import ReinsuranceContract
    from riskmodel import RiskModel

    Distribution = Union[
        "stats.rv_continuous", "TruncatedDistWrapper", "ReinsuranceDistWrapper"
    ]


class GenericAgent:
    def __init__(self):
        self.cash: float = 0
        self.obligations: MutableSequence["Obligation"] = []
        self.operational: bool = True
        self.profits_losses: float = 0
        self.creditor = None
        self.id = -1

    def _pay(self, obligation: "Obligation"):
        """Method to _pay other class instances.
            Accepts:
                Obligation: Type DataDict
            No return value
            Method removes value payed from the agents cash and adds it to recipient agents cash.
            If the recipient is not operational, redirect the payment to the creditor"""
        amount = obligation.amount
        recipient = obligation.recipient
        purpose = obligation.purpose

        if not amount >= 0:
            raise ValueError(
                "Attempting to pay an obligation for a negative ammount - something is wrong"
            )
        # TODO: Think about what happens when paying non-operational firms
        while not recipient.get_operational():
            if isleconfig.verbose:
                print(
                    f"Redirecting payment with purpose {purpose} due to non-operational firm {recipient.id}"
                )
            recipient = recipient.creditor
        if self.get_operational():
            self.cash -= amount
            if purpose is not "dividend":
                self.profits_losses -= amount
            recipient.receive(amount)
        else:
            if isleconfig.verbose:
                print(f"Payment not processed as firm {self.id} is not operational")

    def get_operational(self) -> bool:
        """Method to return boolean of if agent is operational. Only used as check for payments.
            No accepted values
            Returns Boolean"""
        return self.operational

    def iterate(self, time: int):
        raise NotImplementedError(
            "Iterate is not implemented in GenericAgent, should have be overridden"
        )

    def _effect_payments(self, time: int):
        """Method for checking if any payments are due.
            Accepts:
                time: Type Integer
            No return value
            Method checks firms list of obligations to see if ay are due for this time, then pays them. If the firm
            does not have enough cash then it enters illiquity, leaves the market, and matures all contracts."""
        # TODO: don't really want to be reconstructing lists every time (unless the obligations are naturally sorted by
        #  time, in which case this could be done slightly better). Low priority, but something to consider
        due = [item for item in self.obligations if item.due_time <= time]
        self.obligations = [item for item in self.obligations if item.due_time > time]

        sum_due = sum([item.amount for item in due])
        if sum_due > self.cash:
            self.obligations += due
            self.enter_illiquidity(time, sum_due)
        else:
            for obligation in due:
                self._pay(obligation)

    def enter_illiquidity(self, time: int, sum_due: float):
        raise NotImplementedError()

    def receive_obligation(
        self, amount: float, recipient: "GenericAgent", due_time: int, purpose: str
    ):
        """Method for receiving obligations that the firm will have to pay.
                    Accepts:
                        amount: Type integer, how much will be paid
                        recipient: Type Class instance, who will be paid
                        due_time: Type Integer, what time value they will be paid
                        purpose: Type string, why they are being paid
                    No return value
                    Adds obligation (Type DataDict) to list of obligations owed by the firm."""

        obligation = Obligation(
            amount=amount, recipient=recipient, due_time=due_time, purpose=purpose
        )
        self.obligations.append(obligation)

    def receive(self, amount: float):
        """Method to accept cash payments."""
        self.cash += amount
        self.profits_losses += amount


@dataclasses.dataclass
class RiskProperties:
    """Class for holding the properties of an insured risk"""

    risk_factor: float
    value: float
    category: int
    owner: "GenericAgent"

    number_risks: int = 1
    contract: "MetaInsuranceContract" = None
    insurancetype: str = None
    deductible: float = None
    runtime: int = None
    expiration: int = None
    excess_fraction: float = None
    deductible_fraction: float = None
    reinsurance_share: float = None
    periodized_total_premium: float = None
    excess: float = None
    runtime_left: int = None


@dataclasses.dataclass
class AgentProperties:
    """Class for holding the properties of an agent"""

    id: int
    initial_cash: float
    riskmodel_config: Mapping
    norm_premium: float
    profit_target: float
    initial_acceptance_threshold: float
    acceptance_threshold_friction: float
    reinsurance_limit: float
    non_proportional_reinsurance_level: float
    capacity_target_decrement_threshold: float
    capacity_target_increment_threshold: float
    capacity_target_decrement_factor: float
    capacity_target_increment_factor: float
    interest_rate: float


@dataclasses.dataclass
class Obligation:
    """Class for holding the properties of an obligation"""

    amount: float
    recipient: "GenericAgent"
    due_time: int
    purpose: str


class ConstantGen(stats.rv_continuous):
    def _pdf(self, x: float, *args) -> float:
        a = np.float_(x == 0)
        a[a == 1.0] = np.inf
        return a

    def _cdf(self, x: float, *args) -> float:
        return np.float_(x >= 0)

    def _rvs(self, *args) -> Union[np.ndarray, float]:
        if self._size is None:
            return 0.0
        else:
            return np.zeros(shape=self._size)


Constant = ConstantGen(name="constant")


class ReinsuranceProfile:
    """Class for keeping track of the reinsurance that an insurance firm holds

    All reinsurance is assumed to be on open intervals

    regions are tuples, (priority, priority+limit, contract), so the contract covers losses in the region (priority,
    priority + limit)"""

    # TODO: add, remove, explode, get uninsured regions
    def __init__(self, riskmodel: "RiskModel"):
        self.reinsured_regions: MutableSequence[
            SortedList[Tuple[int, int, "ReinsuranceContract"]]
        ]

        self.reinsured_regions = [
            SortedList(key=lambda x: x[0])
            for _ in range(isleconfig.simulation_parameters["no_categories"])
        ]

        # Used for automatically updating the riskmodel when reinsurance is modified
        self.riskmodel = riskmodel

    def add(self, contract: "ReinsuranceContract", value: float) -> None:
        lower_bound: int = contract.deductible
        upper_bound: int = contract.excess
        category = contract.category

        self.reinsured_regions[category].add((lower_bound, upper_bound, contract))
        index = self.reinsured_regions[category].index(
            (lower_bound, upper_bound, contract)
        )

        # Check for overlap with region to the right...
        if (
            index + 1 < len(self.reinsured_regions[category])
            and self.reinsured_regions[category][index + 1][0] < upper_bound
        ):
            raise ValueError(
                "Attempted to add reinsurance overlapping with existing reinsurance \n"
                f"Reinsured regions are {self.reinsured_regions[category]}"
            )

        # ... and to the left
        if index != 0 and self.reinsured_regions[category][index - 1][1] > lower_bound:
            raise ValueError(
                "Attempted to add reinsurance overlapping with existing reinsurance \n"
                f"Reinsured regions are {self.reinsured_regions[category]}"
            )

        self.riskmodel.set_reinsurance_coverage(
            value=value, coverage=self.reinsured_regions[category], category=category
        )

    def remove(self, contract: "ReinsuranceContract", value: float) -> None:
        lower_bound = contract.deductible
        upper_bound = contract.excess
        category = contract.category

        try:
            self.reinsured_regions[category].remove(
                (lower_bound, upper_bound, contract)
            )
        except ValueError:
            raise ValueError(
                "Attempting to remove a reinsurance contract that doesn't exist!"
            )
        self.riskmodel.set_reinsurance_coverage(
            value=value, coverage=self.reinsured_regions[category], category=category
        )

    def uncovered(self, category: int) -> MutableSequence[Tuple[float, float]]:
        uncovered_regions = []
        upper = 0
        for region in self.reinsured_regions[category]:
            if region[0] - upper > 1:
                # There's a gap in coverage!
                uncovered_regions.append((upper, region[0]))
            upper = region[1]
        uncovered_regions.append((upper, np.inf))
        return uncovered_regions

    def contracts_to_explode(
        self, category: int, damage: float
    ) -> MutableSequence["ReinsuranceContract"]:
        contracts = []
        for region in self.reinsured_regions[category]:
            if region[0] < damage:
                contracts.append(region[2])
                if region[1] >= damage:
                    break
        return contracts

    def all_contracts(self) -> MutableSequence["ReinsuranceContract"]:
        regions = chain.from_iterable(self.reinsured_regions)
        contracts = map(lambda x: x[2], regions)
        return list(contracts)

    def update_value(self, value: float, category: int) -> None:
        self.riskmodel.set_reinsurance_coverage(
            value=value, coverage=self.reinsured_regions[category], category=category
        )

    @staticmethod
    def split_longest(
        l: MutableSequence[Tuple[float, float]]
    ) -> MutableSequence[Tuple[float, float]]:
        max_width = 0
        max_width_index = None
        for i, region in enumerate(l):
            if region[1] - region[0] > max_width:
                max_width = region[1] - region[0]
                max_width_index = i
        if max_width == 0:
            raise RuntimeError("All regions have zero width!")
        lower, upper = l[max_width_index]
        mid = (lower + upper) / 2
        del l[max_width_index]
        l.insert(max_width_index, (mid, upper))
        l.insert(max_width_index, (lower, mid))
        return l
