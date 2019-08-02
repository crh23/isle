import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


class CompareData:
    def __init__(self, original_filename, new_filename, extra_filename=None):
        """Initialises the CompareData class. Is provided with two or three filenames and unpacks them, also creating
        dictionaries of the average values in case of ensemble/replication runs.
            Accepts:
                original_filename: Type String.
                new_filename: Type String.
                extra_filename: Type String. Defaults None but there in case of extra file to be compared."""
        with open(original_filename, "r") as rfile:
            self.original_data = [eval(k) for k in rfile]
        with open(new_filename, "r") as rfile:
            self.new_data = [eval(k) for k in rfile]
        if extra_filename is not None:
            with open(extra_filename, "r") as rfile:
                self.extra_data = [eval(k) for k in rfile]
            self.extra = True
        else:
            self.extra = False
            self.extra_data = {}

        self.event_damage = []
        self.event_schedule = []
        self.original_averages = {}
        self.new_averages = {}
        self.extra_averages = {}
        dicts = [self.original_averages, self.new_averages, self.extra_averages]
        datas = [self.original_data, self.new_data, self.extra_data]
        for i in range(len(datas)):
            if self.extra is False and i == 2:
                pass
            else:
                self.init_averages(dicts[i], datas[i])

    def init_averages(self, avg_dict, data_dict):
        """Method that initliases the average value dictionaries for the files. Takes a complete data dict and adds the
           average values to a different dict provided.
            Accepts:
                avg_dict: Type Dict. Initially should be empty.
                data_dict: Type List of data dict. Each element is a data dict containing data from that replication.
            No return values."""
        for data in data_dict:
            for key in data.keys():
                if "firms_cash" in key or key == "market_diffvar" or "riskmodels" in key:
                    pass
                elif key == "individual_contracts" or key == "reinsurance_contracts":
                    avg_contract_per_firm = []
                    for t in range(len(data[key][0])):
                        total_contracts = 0
                        for i in range(len(data[key])):
                            if data[key][i][t] > 0:
                                total_contracts += data[key][i][t]
                        if "re" in key:
                            firm_count = data["total_reinoperational"][t]
                        else:
                            firm_count = data["total_operational"][t]
                        if firm_count > 0:
                            avg_contract_per_firm.append(total_contracts / firm_count)
                        else:
                            avg_contract_per_firm.append(0)
                    if key not in avg_dict.keys():
                        avg_dict[key] = avg_contract_per_firm
                    else:
                        avg_dict[key] = [list1 + list2 for list1, list2 in zip(avg_dict[key], avg_contract_per_firm)]
                elif key == "rc_event_schedule_initial":
                    self.event_schedule.append(data[key])
                elif key == "rc_event_damage_initial":
                    self.event_damage.append(data[key])
                else:
                    if key not in avg_dict.keys():
                        avg_dict[key] = data[key]
                    else:
                        avg_dict[key] = [list1 + list2 for list1, list2 in zip(avg_dict[key], data[key])]
        for key in avg_dict.keys():
            avg_dict[key] = [value/len(data_dict) for value in avg_dict[key]]

    def plot(self, upper, lower, events=False):
        """Method to plot same type of data for different files on a plot.
        No accepted values.
        No return values."""
        for key in self.original_averages.keys():
            plt.figure()
            original_values = self.original_averages[key][lower:upper]
            mean_original_values = np.mean(original_values)
            new_values = self.new_averages[key][lower:upper]
            mean_new_values = np.mean(new_values)
            xvalues = np.arange(lower, upper)
            percent_diff = self.stats(original_values, new_values)
            plt.plot(xvalues, original_values, label='Original Values', color="blue")
            plt.plot(xvalues, new_values, label='New Values, Avg Diff = %f%%' % percent_diff, color="red")
            if self.extra:
                extra_values = self.extra_averages[key][lower:upper]
                mean_extra_values = np.mean(extra_values)
                percent_diff = self.stats(original_values, extra_values)
                plt.plot(xvalues, extra_values, label="Extra Values, Avg Diff = %f%%" % percent_diff, color="yellow")
            if "cum" not in key:
                mean_diff = self.mean_diff(mean_original_values, mean_new_values)
                plt.axhline(mean_original_values, linestyle='--', label="Original Mean",  color="blue")
                plt.axhline(mean_new_values, linestyle='--', label="New Mean, Diff = %f%%" % mean_diff,  color="red")
                if self.extra:
                    mean_diff = self.mean_diff(mean_original_values, mean_extra_values)
                    plt.axhline(mean_extra_values, linestyle='--', label='Extra Mean, Diff = %f%%' % mean_diff, color='yellow')
            if events:
                for categ_index in range(len(self.event_schedule[0])):
                    for event_index in range(len(self.event_schedule[0][categ_index])):
                        if self.event_damage[0][categ_index][event_index] > 0.5:
                            plt.axvline(self.event_schedule[0][categ_index][event_index], linestyle='-',  color='green')
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel(key)
        plt.show()

    def ks_test(self):
        """Method to perform ks test on two sets of file data. Returns the D statistic and p-value.
            No accepted values.
            No return values."""
        for key in self.original_averages.keys():
            original_values = self.original_averages[key]
            new_values = self.new_averages[key]
            D, p = ss.ks_2samp(original_values, new_values)
            print("%s has p value: %f and D: %f" % (key, p, D))

    def chi_squared(self):
        for key in self.original_averages.keys():
            original_values = self.original_averages[key][200:1000]
            new_values = self.new_averages[key][200:1000]
            fractional_diff = 0
            for time in range(len(original_values)):
                if original_values[time] != 0:
                    fractional_diff += (original_values[time] - new_values[time])**2 / original_values[time]
            print("%s has chi squared value: %f" % (key, fractional_diff/len(original_values)))

    def stats(self, original_values, new_values):
        percentage_diff_sum = 0
        for time in range(len(original_values)):
            if original_values[time] != 0:
                percentage_diff_sum += np.absolute((original_values[time]-new_values[time])/original_values[time]) * 100
        return percentage_diff_sum / len(original_values)

    def mean_diff(self, original_mean, new_mean):
        diff = (new_mean - original_mean) / original_mean
        return diff * 100


CD = CompareData("data/single_history_logs_old_2019_Aug_02_12_53.dat",
                 "data/single_history_logs.dat",
                 "data/single_history_logs_old_2019_Aug_02_16_45.dat")
# CD.plot(events=False, upper=1000, lower=200)
CD.chi_squared()