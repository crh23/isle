import isleconfig
from metainsuranceorg import MetaInsuranceOrg
from genericclasses import Obligation, GenericAgent

from typing import Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurancesimulation import InsuranceSimulation
    from metainsurancecontract import MetaInsuranceContract


# noinspection PyAbstractClass
class CatBond(MetaInsuranceOrg):
    # noinspection PyMissingConstructor
    # TODO inheret GenericAgent instead of MetaInsuranceOrg? Or maybe some common root
    def __init__(
        self,
        simulation: "InsuranceSimulation",
        per_period_premium: float,
        owner: GenericAgent,
    ):
        """Initialising methods.
            Accepts:
                simulation: Type class
                per_period_premium: Type decimal
                owner: Type class
        This initialised the catbond class instance, inheriting methods from MetaInsuranceOrg."""
        self.simulation = simulation
        self.simulation_parameters = simulation.simulation_parameters
        self.id: int = self.simulation.get_unique_catbond_id()
        self.underwritten_contracts: Collection["MetaInsuranceContract"] = []
        self.cash: float = 0
        self.profits_losses: float = 0
        self.obligations: Collection[Obligation] = []
        self.operational: bool = True
        self.owner: GenericAgent = owner
        self.per_period_dividend: float = per_period_premium
        self.creditor = self.simulation
        self.expiration: int = None
        # self.simulation_no_risk_categories = self.simulation.simulation_parameters["no_categories"]

    def iterate(self, time: int):
        """Method to perform CatBond duties for each time iteration.
            Accepts:
                time: Type Integer
            No return values
        For each time iteration this is called from insurancesimulation to perform duties: interest payments,
        _pay obligations, mature the contract if ended, make payments."""
        assert len(self.underwritten_contracts) == 1
        # Interest gets paid directly to the owner of the catbond (i.e. the simulation)
        self.simulation.bank.award_interest(self.owner, self.cash)
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
            print("Number of underwritten contracts ", len(self.underwritten_contracts))

        """mature contracts"""
        if self.underwritten_contracts[0].expiration <= time:
            self.underwritten_contracts[0].mature(time)
            self.underwritten_contracts = []
            self.mature_bond()

        else:
            if self.operational:
                self.pay_dividends(time)

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
        self.expiration = contract.expiration

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
            self.obligations = []
            self.simulation.delete_agents([self])
            self.operational = False
        else:
            print("CatBond is not operational so cannot mature")

    def get_available_cash(self, time: int) -> float:
        """Returns the amount of cash the CatBond has available to pay out in claims (i.e. not reserved for premiums).
        Used to update limit on contract"""
        return self.cash - self.per_period_dividend * (self.expiration - time + 1)

    def enter_illiquidity(self, time: int, sum_due: float):

        raise RuntimeError("CatBond has run out of money, that shouldn't happen")
