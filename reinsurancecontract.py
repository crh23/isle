from metainsurancecontract import MetaInsuranceContract


class ReinsuranceContract(MetaInsuranceContract):
    """ReinsuranceContract class.
        Inherits from InsuranceContract.
        Constructor is not currently required but may be used in the future to distinguish InsuranceContract
            and ReinsuranceContract objects.
        The signature of this class' constructor is the same as that of the InsuranceContract constructor.
        The class has two methods (explode, mature) that overwrite methods in InsuranceContract."""

    def __init__(
        self,
        insurer,
        risk,
        time,
        premium,
        runtime,
        payment_period,
        expire_immediately,
        initial_var=0.0,
        insurancetype="proportional",
        deductible_fraction=None,
        excess_fraction=None,
        reinsurance=0,
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
            excess_fraction,
            reinsurance,
        )
        # self.is_reinsurancecontract = True

        if self.insurancetype not in ["excess-of-loss", "proportional"]:
            raise ValueError(f'Unrecognised insurance type "{self.insurancetype}"')
        if self.insurancetype == "excess-of-loss":
            self.property_holder.add_reinsurance(
                category=self.category,
                excess_fraction=self.excess_fraction,
                deductible_fraction=self.deductible_fraction,
                contract=self,
            )
        else:
            assert self.contract is not None

    def explode(self, time, damage_extent=None):
        """Explode method.
               Accepts arguments
                   time: Type integer. The current time.
                   uniform_value: Not used
                   damage_extent: Type float. The absolute damage in excess-of-loss reinsurance (not relative as in 
                                       proportional contracts. 
               No return value.
           Method marks the contract for termination.
            """

        # QUERY: What is the difference? Also, what happens if damage_extent = None?
        if damage_extent > self.deductible:
            # QUERY: Changed this, for the better?
            if self.insurancetype == "excess-of-loss":
                claim = min(self.excess, damage_extent) - self.deductible
                self.insurer.receive_obligation(
                    claim, self.property_holder, time, "claim"
                )
            elif self.insurancetype == "proportional":
                claim = min(self.excess, damage_extent) - self.deductible
                self.insurer.receive_obligation(
                    claim, self.property_holder, time + 1, "claim"
                )
            else:
                raise ValueError(f"Unexpected insurance type {self.insurancetype}")
                # Reinsurer pays as soon as possible.
            # Every reinsurance claim made is immediately registered.
            self.insurer.register_claim(claim)

            if self.expire_immediately:
                self.current_claim += self.contract.claim
                # TODO: should proportional reinsurance claims be subject to excess_of_loss retrocession?
                #  If so, reorganize more straightforwardly

                self.expiration = time
                # self.terminating = True

    def mature(self, time):
        """Mature method. 
               Accepts arguments
                    time: Type integer. The current time.
               No return value.
           Removes any reinsurance functions this contract has and terminates any reinsurance contracts for this 
           contract."""
        # self.terminating = True
        self.terminate_reinsurance(time)

        if self.insurancetype == "excess-of-loss":
            self.property_holder.delete_reinsurance(
                category=self.category, contract=self
            )
        else:  # TODO: ? Instead: if self.insurancetype == "proportional":
            self.contract.unreinsure()
