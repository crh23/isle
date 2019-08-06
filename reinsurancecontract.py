import metainsurancecontract

from typing import Optional
from typing import TYPE_CHECKING
from math import floor

if TYPE_CHECKING:
    from insurancefirms import InsuranceFirm
    from metainsuranceorg import MetaInsuranceOrg
    from genericclasses import RiskProperties
    from catbond import Catbond


class ReinsuranceContract(metainsurancecontract.MetaInsuranceContract):
    """ReinsuranceContract class.
        Inherits from InsuranceContract.
        Constructor is not currently required but may be used in the future to distinguish InsuranceContract
            and ReinsuranceContract objects.
        The signature of this class' constructor is the same as that of the InsuranceContract constructor.
        The class has two methods (explode, mature) that overwrite methods in InsuranceContract."""

    def __init__(
        self,
        insurer: "MetaInsuranceOrg",
        risk: "RiskProperties",
        time: int,
        per_value_premium: float,
        runtime: int,
        payment_period: int,
        expire_immediately: bool,
        initial_var: float = 0.0,
        insurancetype: str = "proportional",
        deductible_fraction: "Optional[float]" = None,
        limit_fraction: "Optional[float]" = None,
        reinsurance: float = 0,
    ):
        super().__init__(
            insurer,
            risk,
            time,
            per_value_premium,
            runtime,
            payment_period,
            expire_immediately,
            initial_var,
            insurancetype,
            deductible_fraction,
            limit_fraction,
            reinsurance,
        )
        # self.is_reinsurancecontract = True
        self.property_holder: "InsuranceFirm"
        if self.insurancetype not in ["excess-of-loss", "proportional"]:
            raise ValueError(f'Unrecognised insurance type "{self.insurancetype}"')
        if self.insurancetype == "excess-of-loss":
            self.property_holder.add_reinsurance(contract=self)
        else:
            assert self.contract is not None

    def explode(
        self, time: int, uniform_value: None = None, damage_extent: float = None
    ):
        """Explode method.
               Accepts arguments
                   time: Type integer. The current time.
                   uniform_value: Not used
                   damage_extent: Type float. The absolute damage in excess-of-loss reinsurance (not relative as in
                                       proportional contracts.
               No return value.
           Method marks the contract for termination.
            """
        # Just a type hint since for a generic insurance contract property_holder can be the simulation
        self.property_holder: "InsuranceFirm"

        assert uniform_value is None
        if damage_extent is None:
            raise ValueError("Damage extent should be given")
        if damage_extent > self.deductible:
            # Proportional reinsurance is triggered by the individual reinsured contracts at the time of explosion.
            # Since EoL reinsurance isn't triggered until the insurer manually makes a claim, this would mean that
            # proportional reinsurance pays out a turn earlier than EoL. As such, proportional insurance claims are
            # delayed for 1 turn.
            if self.insurancetype == "excess-of-loss":
                claim = min(self.limit, damage_extent) - self.deductible
                self.insurer.receive_obligation(
                    claim, self.property_holder, time, "claim"
                )
            elif self.insurancetype == "proportional":
                claim = min(self.limit, damage_extent) - self.deductible
                self.insurer.receive_obligation(
                    claim, self.property_holder, time + 1, "claim"
                )
            else:
                raise ValueError(f"Unexpected insurance type {self.insurancetype}")
            # Every reinsurance claim made is immediately registered.
            self.insurer.register_claim(claim)

            if self.expire_immediately:
                self.current_claim += self.contract.claim
                # TODO: should proportional reinsurance claims be subject to excess_of_loss retrocession?
                #  If so, reorganize more straightforwardly

                self.expiration = time
                # self.terminating = True
            elif type(self.insurer).__name__ == "CatBond":
                # Catbonds can only pay out a certain amount in their lifetime, so we update the reinsurance coverage
                # for the issuer
                # TODO: Allow for catbonds that can pay out multiple times?
                self.insurer: "Catbond"
                remaining_cb_cash = self.insurer.get_available_cash(time) - claim
                assert remaining_cb_cash >= 0
                if remaining_cb_cash < 2:
                    # If the claim uses up all the catbond's remaining money, the contract ends
                    self.expiration = time
                elif remaining_cb_cash < self.limit - self.deductible:
                    # If the claim uses up enough money that the catbond can't pay out the full exposure, update the
                    # contract to reflect that
                    self.property_holder.delete_reinsurance(contract=self)
                    self.limit = self.deductible + remaining_cb_cash
                    self.limit = floor(self.limit)
                    self.limit_fraction = self.limit / self.value
                    self.property_holder.add_reinsurance(
                        contract=self, force_value=self.value
                    )
                else:
                    # If the catbond still has enough money to pay out the full exposure, no need to change anything.
                    pass

    def mature(self, time: int):
        """Mature method.
               Accepts arguments
                    time: Type integer. The current time.
               No return value.
           Removes any reinsurance functions this contract has and terminates any reinsurance contracts for this
           contract."""
        # Just a type hint since for a generic insurance contract property_holder can be the simulation
        self.property_holder: "InsuranceFirm"

        self.terminate_reinsurance(time)

        if self.insurancetype == "excess-of-loss":
            self.property_holder.delete_reinsurance(contract=self)
        else:
            self.contract.unreinsure()
