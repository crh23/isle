from isleconfig import simulation_parameters


class CentralBank:
    def __init__(self):
        """Constructor Method.
            No accepted arguments.
        Constructs the CentralBank class. This class is currently only used to award interest payments."""
        self.interest_rate = simulation_parameters["interest_rate"]
        self.inflation_target = 0.02
        self.actual_inflation = 0
        self.onemonth_CPI = 0
        self.twelvemonth_CPI = 0
        self.feedback_counter = 0
        self.prices_list = []

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

    def award_interest(self, firm, total_cash):
        """Method to award interest.
            Accepts:
                firm: Type class, the agent that is to be awarded interest.
                total_cash: Type decimal
        This method takes an agents cash and awards it an interest payment on the cash."""
        interest_payment = total_cash * self.interest_rate
        firm.receive(interest_payment)

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
            self.onemonth_CPI = (
                current_price - self.prices_list[-2]
            ) / self.prices_list[-2]
            self.twelvemonth_CPI = (
                current_price - self.prices_list[-13]
            ) / self.prices_list[-13]
            self.actual_inflation = self.twelvemonth_CPI
