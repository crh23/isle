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
        self.aid_budget = 1000000

    def update_money_supply(self, amount, reduce=True):
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

    def regulate(self, firm_id, firm_cash, firm_var):
        """Method to regulate firms. Checks if their average cash over the last year can cover their average VaR and to
        what percent. Based on solvency 2 which has a MCR of 85% and SCR of 99.5%.
            Accepts:
                firm_id: Type Integer. Firms unique ID.
                firm_cash: Type list of decimals. List of cash for last twelve periods.
                firm_var: Type list of decimals. List of VaR for last twelve periods.
            Returns:
                Type String: "Good", "Warning", "LoseControl".
        This method takes a firms average cash and VaR. If average cash above SCR of 99.5% of VaR then all is well,
        if cash is between 85% and 99.5% then is issued a warning (limits business heavily), if under 85% then firm is
        sold."""
        if firm_id not in self.warnings.keys():
            self.warnings[firm_id] = 0

        avg_firm_cash = np.mean(firm_cash)
        avg_var = np.mean(firm_var)

        if avg_firm_cash >= 0.995 * avg_var:
            self.warnings[firm_id] = 0
        elif avg_firm_cash >= 0.85 * avg_var:
            self.warnings[firm_id] += 1
        elif avg_firm_cash < 0.85* avg_var:
            if self.warnings[firm_id] > 0:
                self.warnings[firm_id] = 2
            else:
                self.warnings[firm_id] += 1

        if self.warnings[firm_id] == 0:
            return "Good"
        elif self.warnings[firm_id] == 1:
            return "Warning"
        elif self.warnings[firm_id] >= 2:
            return "LoseControl"

    def adjust_aid_budget(self, time):
        if time % 12 == 0:
            money_left = self.aid_budget
            self.aid_budget = 1000000
            money_taken = self.aid_budget - money_left

    def provide_aid(self, insurance_firms, damage_fraction, time):
        all_firms_aid = 0
        given_aid_dict = {}
        if damage_fraction > 0.50:
            for insurer in insurance_firms:
                claims = sum([ob['amount'] for ob in insurer.obligations if ob["purpose"] == "claim" and ob["due_time"] == time + 2])
                aid = claims * damage_fraction
                all_firms_aid += aid
                given_aid_dict[insurer] = aid
            for fraction in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
                if self.aid_budget - (all_firms_aid * fraction) > 0:
                    self.aid_budget -= (all_firms_aid * fraction)
                    for key in given_aid_dict:
                        given_aid_dict[key] *= fraction
                    print("Damage %f causes %d to be given out in aid. %d budget left." % (damage_fraction, all_firms_aid * fraction, self.aid_budget))
                    return given_aid_dict
        else:
            return given_aid_dict
