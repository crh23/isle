import isleconfig
from metainsuranceorg import MetaInsuranceOrg
from genericclasses import Obligation, GenericAgent

from typing import MutableSequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurancesimulation import InsuranceSimulation
    from metainsurancecontract import MetaInsuranceContract


# TODO: This and MetaInsuranceOrg should probably both subclass something simple - a MetaAgent, say. MetaInsuranceOrg
#  can do more than a CatBond should be able to!


# noinspection PyAbstractClass
class CatBond(MetaInsuranceOrg):
    # noinspection PyMissingConstructor
    # TODO inheret GenericAgent instead of MetaInsuranceOrg?
    def __init__(
        self,
        simulation: "InsuranceSimulation",
        per_period_premium: float,
        owner: GenericAgent,
        interest_rate: float = 0,
    ):
        """Initialising methods.
            Accepts:
                simulation: Type class
                per_period_premium: Type decimal
                owner: Type class
        This initialised the catbond class instance, inheriting methods from MetaInsuranceOrg."""
        self.simulation = simulation
        self.id: int = 0
        self.underwritten_contracts: MutableSequence["MetaInsuranceContract"] = []
        self.cash: float = 0
        self.profits_losses: float = 0
        self.obligations: MutableSequence[Obligation] = []
        self.operational: bool = True
        self.owner: GenericAgent = owner
        self.per_period_dividend: float = per_period_premium
        self.interest_rate: float = interest_rate
        # TODO: shift obtain_yield method to insurancesimulation, thereby making it unnecessary to drag parameters like
        #  self.interest_rate from instance to instance and from class to class
        # self.simulation_no_risk_categories = self.simulation.simulation_parameters["no_categories"]

    # TODO: change start and InsuranceSimulation so that it iterates CatBonds
    def iterate(self, time: int):
        """Method to perform CatBond duties for each time iteration.
            Accepts:
                time: Type Integer
            No return values
        For each time iteration this is called from insurancesimulation to perform duties: interest payments,
        _pay obligations, mature the contract if ended, make payments."""
        self.obtain_yield(time)
        self._effect_payments(time)
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

        """effect payments from contracts"""
        for contract in self.underwritten_contracts:
            contract.check_payment_due(time)

        if not self.underwritten_contracts:
            # If there are no contracts left, the bond is matured
            self.mature_bond()  # TODO: mature_bond method should check if operational

        # TODO: dividend should only be payed according to pre-arranged schedule,
        #  and only if no risk events have materialized so far
        else:
            if self.operational:
                self.pay_dividends(time)

        # self.estimate_var()   # cannot compute VaR for catbond as catbond does not have a riskmodel

    def set_owner(self, owner: GenericAgent):
        """Method to set owner of the Cat Bond.
            Accepts:
                owner: Type class
            No return values."""
        self.owner = owner
        if isleconfig.verbose:
            print("SOLD")

    def set_contract(self, contract: "MetaInsuranceContract"):
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
            obligation = Obligation(
                amount=self.cash,
                recipient=self.simulation,
                due_time=1,
                purpose="mature",
            )
            self._pay(obligation)
            self.simulation.delete_agents([self])
            self.operational = False
        else:
            print("CatBond is not operational so cannot mature")
