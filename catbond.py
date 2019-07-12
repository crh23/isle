import isleconfig
import numpy as np
import scipy.stats
from insurancecontract import InsuranceContract
from reinsurancecontract import ReinsuranceContract
from metainsuranceorg import MetaInsuranceOrg
from riskmodel import RiskModel
import sys, pdb
import uuid


class CatBond(MetaInsuranceOrg):
    def __init__(self, simulation, per_period_premium, owner):
        """Initialising methods.
            Accepts:
                simulation: Type class
                per_period_premium: Type decimal
                owner: Type class
        This initialised the catbond class instance, inheriting methods from MetaInsuranceOrg."""
        self.simulation = simulation
        self.id = 0
        self.underwritten_contracts = []
        self.cash = 0
        self.profits_losses = 0
        self.obligations = []
        self.operational = True
        self.owner = owner
        self.per_period_dividend = per_period_premium

    def iterate(self, time):
        """Method to perform CatBond duties for each time iteration.
            Accepts:
                time: Type Integer
            No return values
        For each time iteration this is called from insurancesimulation to perform duties: interest payments,
        pay obligations, mature the contract if ended, make payments."""
        self.simulation.bank.award_interest(self, self.cash)
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

            """mature contracts"""
            print("Number of underwritten contracts ", len(self.underwritten_contracts))
        maturing = [
            contract
            for contract in self.underwritten_contracts
            if contract.expiration <= time
        ]
        for contract in maturing:
            self.underwritten_contracts.remove(contract)
            contract.mature(time)

        [contract.check_payment_due(time) for contract in self.underwritten_contracts]

        if self.underwritten_contracts == []:
            self.mature_bond()
        else:  # TODO: dividend should only be payed according to pre-arranged schedule, and only if no risk events have materialized so far
            if self.operational:
                self.pay_dividends(time)

    def set_owner(self, owner):
        """Method to set owner of the Cat Bond.
            Accepts:
                owner: Type class
            No return values."""
        self.owner = owner
        if isleconfig.verbose:
            print("SOLD")

    def set_contract(self, contract):
        """Method to record new instances of CatBonds.
            Accepts:
                owner: Type class
            No return values
        Only one contract is ever added to the list of underwritten contracts as each CatBond is a contract itself."""
        self.underwritten_contracts.append(contract)

    def mature_bond(self):
        """Method to mature CatBond.
            No accepted values
            No return values
        When the catbond contract matures this is called which pays the value of the catbond to the simulation, and is
        then deleted from the list of agents."""
        if self.operational:
            obligation = {
                "amount": self.cash,
                "recipient": self.simulation,
                "due_time": 1,
                "purpose": "mature",
            }
            self.pay(obligation)
            self.simulation.delete_agents("catbond", [self])
            self.operational = False
        else:
            print("CatBond is not operational so cannot mature")
