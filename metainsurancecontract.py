from __future__ import annotations
from genericclasses import RiskProperties
import metainsuranceorg


class MetaInsuranceContract:
    def __init__(
        self,
        insurer: metainsuranceorg.MetaInsuranceOrg,
        risk: RiskProperties,
        time: int,
        premium: float,
        runtime: int,
        payment_period: int,
        expire_immediately: bool,
        initial_var: float = 0.0,
        insurancetype: str = "proportional",
        deductible_fraction: float = None,
        excess_fraction: float = None,
        reinsurance: float = 0,
    ):
        """Constructor method.
               Accepts arguments
                    insurer: Type InsuranceFirm. 
                    risk: Type RiskProperties.
                    time: Type integer. The current time.
                    premium: Type float.
                    runtime: Type integer.
                    payment_period: Type integer.
                    expire_immediately: Type boolean. True if the contract expires with the first risk event. False
                                       if multiple risk events are covered.
                    initial_var: Type float. Initial value at risk. Used only to compute true and estimated VaR.
                optional:
                    insurancetype: Type string. The type of this contract, especially "proportional" vs "excess_of_loss"
                    deductible_fraction: Type float (or int)
                    excess_fraction: Type float (or int or None)
                    reinsurance: Type float (or int). The value that is being reinsured.
               Returns InsuranceContract.
           Creates InsuranceContract, saves parameters. Creates obligation for premium payment. Includes contract
           in reinsurance network if applicable (e.g. if this is a ReinsuranceContract)."""
        # TODO: argument reinsurance seems senseless; remove?

        # Save parameters
        self.insurer = insurer
        self.risk_factor = risk.risk_factor
        self.category = risk.category
        self.property_holder = risk.owner
        self.value = risk.value
        self.contract = risk.contract  # May be None
        self.risk = risk
        self.insurancetype = (
            risk.insurancetype if insurancetype is None else insurancetype
        )
        self.runtime = runtime
        self.starttime = time
        self.expiration = runtime + time
        self.expire_immediately = expire_immediately
        self.terminating = False
        self.current_claim = 0
        self.initial_VaR = initial_var
        # set deductible from argument, risk property or default value, whichever first is not None
        default_deductible_fraction = 0.0
        self.deductible_fraction = (
            deductible_fraction
            if deductible_fraction is not None
            else risk.deductible_fraction
            if risk.deductible_fraction is not None
            else default_deductible_fraction
        )

        self.deductible = self.deductible_fraction * self.value

        # set excess from argument, risk property or default value, whichever first is not None
        default_excess_fraction = 1.0
        self.excess_fraction = (
            excess_fraction
            if excess_fraction is not None
            else risk.excess_fraction
            if risk.excess_fraction is not None
            else default_excess_fraction
        )

        self.excess = self.excess_fraction * self.value

        self.reinsurance = reinsurance
        self.reinsurer = None
        self.reincontract = None
        self.reinsurance_share = None

        # setup payment schedule
        # TODO: excess and deductible should not be considered linearily in premium computation; this should be
        #  shifted to the (re)insurer who supplies the premium as argument to the contract's constructor method
        total_premium = premium * self.value
        self.periodized_premium = total_premium / self.runtime

        # N.B.: payment times and values are in reverse, so the earliest time is at the end! This is because popping
        # items off the end of lists is much easier than popping them off the start.
        self.payment_times = [
            time + i for i in range(runtime - 1, -1, -1) if i % payment_period == 0
        ]

        self.payment_values = [total_premium / len(self.payment_times)] * len(
            self.payment_times
        )

        # Embed contract in reinsurance network, if applicable
        if self.contract:
            self.contract.reinsure(
                reinsurer=self.insurer,
                reinsurance_share=risk.reinsurance_share,
                reincontract=self,
            )

        # This flag is set to 1, when the contract is about to expire and there is an attempt to roll it over.
        self.roll_over_flag = 0

    def check_payment_due(self, time: int):
        """Method to check if a contract payment is due.
                    Accepts:
                        time: Type integer
                    No return values.
                This method checks if a scheduled premium payment is due, pays it to the insurer,
                    and removes from schedule."""
        if len(self.payment_times) > 0 and time >= self.payment_times[-1]:
            # Create obligation for premium payment
            self.property_holder.receive_obligation(
                self.payment_values[-1], self.insurer, time, "premium"
            )

            # Remove current payment from payment schedule
            del self.payment_times[-1]
            del self.payment_values[-1]

    def get_and_reset_current_claim(self):
        """Method to return and reset claim.
            No accepted values
            Returns:
                self.category: Type integer. Which category the contracted risk is in.
                current_claim: Type decimal
                self.insurancetype == "proportional": Type Boolean. Returns True if insurance is
                    proportional and vice versa.
        This method retuns the current claim, then resets it, and also indicates the type of insurance."""
        current_claim = self.current_claim
        self.current_claim = 0
        return self.category, current_claim, (self.insurancetype == "proportional")

    def terminate_reinsurance(self, time: int):
        """Terminate reinsurance method.
               Accepts arguments
                    time: Type integer. The current time.
               No return value.
           Causes any reinsurance contracts to be dissolved as the present contract terminates."""
        if self.reincontract is not None:
            self.reincontract.dissolve(time)

    def dissolve(self, time: int):
        """Dissolve method.
               Accepts arguments
                    time: Type integer. The current time.
               No return value.
            Marks the contract as terminating (to avoid new ReinsuranceContracts for this contract)."""
        self.expiration = time

    def reinsure(self, reinsurer, reinsurance_share, reincontract):
        """Reinsure Method.
               Accepts arguments:
                   reinsurer: Type ReinsuranceFirm. The reinsurer.
                   reinsurance_share: Type float. Share of the value that is proportionally reinsured.
                   reincontract: Type ReinsuranceContract. The reinsurance contract.
               No return value.
           Adds parameters for reinsurance of the current contract."""
        self.reinsurer = reinsurer
        self.reinsurance = self.value * reinsurance_share
        self.reinsurance_share = reinsurance_share
        self.reincontract = reincontract
        assert self.reinsurance_share in [None, 0.0, 1.0]

    def unreinsure(self):
        """Unreinsurance Method.
               Accepts no arguments:
               No return value.
           Removes parameters for reinsurance of the current contract. To be called when reinsurance has terminated."""
        self.reinsurer = None
        self.reincontract = None
        self.reinsurance = 0
        self.reinsurance_share = None

    def explode(self, time, uniform_value=None, damage_extent=None):
        """Explode method.
               Accepts arguments
                   time: Type integer. The current time.
                   uniform_value: Not used
                   damage_extent: Type float. The absolute damage in excess-of-loss reinsurance (not relative as in
                                       proportional contracts.
               No return value.
           Method marks the contract for termination.
            """
        raise NotImplementedError()

    def mature(self, time):
        """Mature method.
               Accepts arguments
                    time: Type integer. The current time.
               No return value.
           Removes any reinsurance functions this contract has and terminates any reinsurance contracts for this
           contract."""
        raise NotImplementedError()
