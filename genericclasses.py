from __future__ import annotations
import dataclasses
from typing import Mapping
import metainsurancecontract


class GenericAgent:
    def __init__(self):
        self.cash = 0
        self.obligations = []
        self.operational = True
        self.profits_losses = 0

    def _pay(self, obligation):
        """Method to _pay other class instances.
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

    def get_operational(self):
        """Method to return boolean of if agent is operational. Only used as check for payments.
            No accepted values
            Returns Boolean"""
        return self.operational

    def iterate(self, time):
        raise NotImplementedError(
            "Iterate is not implemented in GenericAgent, should have be overridden"
        )

    def receive_obligation(self, amount, recipient, due_time, purpose):
        """Method for receiving obligations that the firm will have to _pay.
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
    owner: GenericAgent

    number_risks: int = 1
    contract: metainsurancecontract.MetaInsuranceContract = None
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
