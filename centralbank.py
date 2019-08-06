from isleconfig import simulation_parameters
import numpy as np


class CentralBank:
    def __init__(self, money_supply):
        """Constructor Method.
            No accepted arguments.
        Constructs the CentralBank class. This class is currently only used to award interest payments."""
        self.interest_rate = simulation_parameters['interest_rate']
        self.inflation_target = 0.02
        self.actual_inflation = 0
        self.onemonth_CPI = 0
        self.twelvemonth_CPI = 0
        self.feedback_counter = 0
        self.prices_list = []
        self.economy_money = money_supply
        self.warnings = {}
        self.aid_budget = self.aid_budget_reset = simulation_parameters['aid_budget']

    def update_money_supply(self, amount, reduce=True):
        """Method to update the current supply of money in the insurance simulation economy. Only used to monitor
        supply, all handling of money (e.g obligations) is done by simulation.
            Accepts:
                amount: Type Integer.
                reduce: Type Boolean."""
        if reduce:
            self.economy_money -= amount
        else:
            self.economy_money += amount
        assert self.economy_money > 0

    def award_interest(self, firm, total_cash):
        """Method to award interest.
            Accepts:
                firm: Type class, the agent that is to be awarded interest.
                total_cash: Type decimal
        This method takes an agents cash and awards it an interest payment on the cash."""
        interest_payment = total_cash * self.interest_rate
        firm.receive(interest_payment)
        self.update_money_supply(interest_payment, reduce=True)

    def set_interest_rate(self):
        """Method to set the interest rate
            No accepted arguments
            No return values
        This method is meant to set interest rates dependant on prices however insurance firms have little effect on
        interest rates therefore is not used and needs work if to be used."""
        if self.actual_inflation > self.inflation_target:
            if self.feedback_counter > 4:
                self.interest_rate += 0.0001
                self.feedback_counter = 0
            else:
                self.feedback_counter += 1
        elif self.actual_inflation < -0.01:
            if self.feedback_counter > 4:
                if self.interest_rate > 0.0001:
                    self.interest_rate -= 0.0001
                    self.feedback_counter = 0
            else:
                self.feedback_counter += 1
        else:
            self.feedback_counter = 0
        print(self.interest_rate)

    def calculate_inflation(self, current_price, time):
        """Method to calculate inflation in insurance prices.
            Accepts:
                current_price: Type decimal
                time: Type integer
        This method is designed to calculate both the percentage change in insurance price last 1 and 12 months as an
        estimate of inflation. This is to help calculate how insurance rates should be set. Currently unused."""
        self.prices_list.append(current_price)
        if time < 13:
            self.actual_inflation = self.inflation_target
        else:
            self.onemonth_CPI = (current_price - self.prices_list[-2])/self.prices_list[-2]
            self.twelvemonth_CPI = (current_price - self.prices_list[-13])/self.prices_list[-13]
            self.actual_inflation = self.twelvemonth_CPI

    def regulate(self, firm_id, firm_cash, firm_var, reinsurance, age, safety_margin):
        """Method to regulate firms
            Accepts:
                firm_id: Type Integer. Firms unique ID.
                firm_cash: Type list of decimals. List of cash for last twelve periods.
                firm_var: Type list of decimals. List of VaR for last twelve periods.
                reinsurance: Type List of Lists of Lists. Contains deductible and excess values for each reinsurance
                             contract in each category for each iteration.
                age: Type Integer.
            Returns:
                Type String: "Good", "Warning", "LoseControl".
        This method calculates how much each reinsurance contract would pay out if all VaR in respective category was
        claimed, adds to cash for that iteration and calculated fraction of capital to total VaR. If average fraction
        over all iterations is above SCR (from solvency ii) of 99.5% of VaR then all is well, if cash is between 85% and
        99.5% then is issued a warning (limits business heavily), if under 85% then firm is sold. Each firm is given
        initial 24 iteration period that it cannot lose control otherwise all firm immediately bankrupt."""
        if firm_id not in self.warnings.keys():
            self.warnings[firm_id] = 0

        # Calculates reinsurance that covers VaR for each category in each iteration and adds to cash.
        cash_fractions = []
        for iter in range(len(reinsurance)):
            reinsurance_capital = 0
            for categ in range(len(reinsurance[iter])):
                for contract in reinsurance[iter][categ]:
                    if firm_var[iter][categ] / safety_margin >= contract[0]:  # Check VaR greater than deductible
                        if firm_var[iter][categ] / safety_margin >= contract[1]:  # Check VaR greater than excess
                            reinsurance_capital += (contract[1] - contract[0])
                        else:
                            reinsurance_capital += (firm_var[iter][categ] - contract[0])
                    else:
                        reinsurance_capital += 0  # If below deductible no reinsurance
            if sum(firm_var[iter]) > 0:
                cash_fractions.append((firm_cash[iter] + reinsurance_capital) / sum(firm_var[iter]))
            else:
                cash_fractions.append(1)

        avg_var_coverage = safety_margin * np.mean(cash_fractions)  # VaR contains margin of safety (=2x) not actual value

        if avg_var_coverage >= 0.995:
            self.warnings[firm_id] = 0
        elif avg_var_coverage >= 0.85:
            self.warnings[firm_id] += 1
        elif avg_var_coverage < 0.85:
            if age < 24:
                self.warnings[firm_id] += 1
            else:
                self.warnings[firm_id] = 2

        if self.warnings[firm_id] == 0:
            return "Good"
        elif self.warnings[firm_id] == 1:
            return "Warning"
        elif self.warnings[firm_id] >= 2:
            return "LoseControl"

    def adjust_aid_budget(self, time):
        """Method to reset the aid budget every 12 iterations (i.e. a year)
            Accepts:
                time: type Integer.
            No return values."""
        if time % 12 == 0:
            money_left = self.aid_budget
            self.aid_budget = self.aid_budget_reset
            money_taken = self.aid_budget - money_left

    def provide_aid(self, insurance_firms, damage_fraction, time):
        """Method to provide aid to firms if enough damage.
            Accepts:
                insurance_firms: Type List of Classes.
                damage_fraction: Type Decimal.
                time: Type Integer.
            Returns:
                given_aid_dict: Type DataDict. Each key is an insurance firm with the value as the aid provided.
        If damage is above a given threshold then firms are given a percentage of total claims as aid (as cannot provide
        actual policyholders with cash) based on damage fraction and how much budget is left. Each firm given equal
        proportion. Returns data dict of values so simulation instance can pay."""
        all_firms_aid = 0
        given_aid_dict = {}
        if damage_fraction > 0.50:
            for insurer in insurance_firms:
                claims = sum([ob['amount'] for ob in insurer.obligations if ob["purpose"] == "claim" and ob["due_time"] == time + 2])
                aid = claims * damage_fraction
                all_firms_aid += aid
                given_aid_dict[insurer] = aid
            # Give each firm an equal fraction of claims
            fractions = np.arange(0, 1.05, 0.05)[::-1]
            for fraction in fractions:
                if self.aid_budget - (all_firms_aid * fraction) > 0:
                    self.aid_budget -= (all_firms_aid * fraction)
                    for key in given_aid_dict:
                        given_aid_dict[key] *= fraction
                    print("Damage %f causes %d to be given out in aid. %d budget left." % (damage_fraction, all_firms_aid * fraction, self.aid_budget))
                    return given_aid_dict
        else:
            return given_aid_dict
