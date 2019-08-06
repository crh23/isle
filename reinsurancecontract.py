import metainsurancecontract

from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurancefirms import InsuranceFirm
    from metainsuranceorg import MetaInsuranceOrg
    from genericclasses import RiskProperties


class ReinsuranceContract(metainsurancecontract.MetaInsuranceContract):
    """ReinsuranceContract class.
        Inherits from InsuranceContract.
        The signature of this class' constructor is the same as that of the InsuranceContract constructor.
        The class has two methods (explode, mature) that overwrite methods in InsuranceContract."""

    def __init__(self,insurer: "MetaInsuranceOrg", risk: "RiskProperties", time: int, premium: float, runtime: int,
                 payment_period: int, expire_immediately: bool, initial_var: float = 0.0,
                 insurancetype: str = "proportional", deductible_fraction: "Optional[float]" = None,
                 limit_fraction: "Optional[float]" = None, reinsurance: float = 0,):
        super().__init__(
            insurer,
            risk,
            time,
            premium,
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

    def explode(self, time: int, uniform_value: None = None, damage_extent: float = None):
        """Explode method.
               Accepts arguments
                   time: Type integer. The current time.
                   uniform_value: Not used
                   damage_extent: Type float. The absolute damage in excess-of-loss reinsurance (not relative as in
                                       proportional contracts.
               No return value.
           Method marks the contract for termination.
            """
        assert uniform_value is None
        if damage_extent is None:
            raise ValueError("Damage extend should be given")
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

    def mature(self, time: int):
        """Mature method.
               Accepts arguments
                    time: Type integer. The current time.
               No return value.
           Removes any reinsurance functions this contract has and terminates any reinsurance contracts for this
           contract."""
        # self.terminating = True
        self.terminate_reinsurance(time)

        if self.insurancetype == "excess-of-loss":
            self.property_holder.delete_reinsurance(contract=self)
        else:  # TODO: ? Instead: if self.insurancetype == "proportional":
            self.contract.unreinsure()
