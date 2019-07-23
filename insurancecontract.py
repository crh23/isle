import metainsurancecontract

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metainsuranceorg import MetaInsuranceOrg
    from insurancesimulation import InsuranceSimulation
    from genericclasses import RiskProperties


class InsuranceContract(metainsurancecontract.MetaInsuranceContract):
    """ReinsuranceContract class.
        Inherits from MetaInsuranceContract.
        Constructor is not currently required but may be used in the future to distinguish InsuranceContract
            and ReinsuranceContract objects.
        The signature of this class' constructor is the same as that of the InsuranceContract constructor.
        The class has two methods (explode, mature) that overwrite methods in InsuranceContract."""

    def __init__(
        self,
        insurer: "MetaInsuranceOrg",
        risk: "RiskProperties",
        time: int,
        premium: float,
        runtime: int,
        payment_period: int,
        expire_immediately: bool,
        initial_var: float = 0.0,
        insurancetype: str = "proportional",
        deductible_fraction: float = None,
        limit_fraction: float = None,
        reinsurance: float = 0,
    ):
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
        # the property holder in an insurance contract should always be the simulation
        assert self.property_holder is self.insurer.simulation
        self.property_holder: "InsuranceSimulation"

    def explode(self, time, uniform_value=None, damage_extent=None):
        """Explode method.
               Accepts arguments
                   time: Type integer. The current time.
                   uniform_value: Type float. Random value drawn in InsuranceSimulation. To determine if this risk
                                  is affected by peril.
                   damage_extent: Type float. Random value drawn in InsuranceSimulation. To determine the extent of
                                  damage caused in the risk insured by this contract.
               No return value.
        For registering damage and creating resulting claims (and payment obligations)."""
        if uniform_value is None:
            raise ValueError(
                "uniform_value must be passed to InsuranceContract.explode"
            )
        if damage_extent is None:
            raise ValueError(
                "damage_extent must be passed to InsuranceContract.explode"
            )
        if uniform_value < self.risk_factor:
            claim = min(self.limit, damage_extent * self.value) - self.deductible
            self.insurer.register_claim(
                claim
            )  # Every insurance claim made is immediately registered.

            self.current_claim += claim
            self.insurer.receive_obligation(
                claim, self.property_holder, time + 2, "claim"
            )
            # Insurer pays one time step after reinsurer to avoid bankruptcy.
            # TODO: Is this realistic? Change this?
            if self.expire_immediately:
                self.expiration = time
                # self.terminating = True

    def mature(self, time):
        """Mature method.
               Accepts arguments
                    time: Type integer. The current time.
               No return value.
           Returns risk to simulation as contract terminates. Calls terminate_reinsurance to dissolve any reinsurance
           contracts."""
        # self.terminating = True

        self.terminate_reinsurance(time)

        if not self.roll_over_flag:
            self.property_holder.return_risks([self.risk])
