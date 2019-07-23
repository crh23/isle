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
        "stats.rv_continuous",
        "TruncatedDistWrapper",
        "ReinsuranceDistWrapper",
    ]


class GenericAgent:
    def __init__(self):
        self.cash: float = 0
        self.obligations: MutableSequence["Obligation"] = []
        self.operational: bool = True
        self.profits_losses: float = 0

    def _pay(self, obligation: "Obligation"):
        """Method to _pay other class instances.
            Accepts:
                Obligation: Type DataDict
            No return value
            Method removes value payed from the agents cash and adds it to recipient agents cash."""
        amount = obligation.amount
        recipient = obligation.recipient
        purpose = obligation.purpose
        # TODO: Think about what happens when paying non-operational firms
        if self.get_operational() and recipient.get_operational():
            self.cash -= amount
            if purpose is not "dividend":
                self.profits_losses -= amount
            recipient.receive(amount)

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
        """Method for receiving obligations that the firm will have to _pay.
                    Accepts:
                        amount: Type integer, how much will be payed
                        recipient: Type Class instance, who will be payed
                        due_time: Type Integer, what time value they will be payed
                        purpose: Type string, why they are being payed
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
