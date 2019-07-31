import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


def OpenAndPlot():
    # Old data with bought bankruptcies and regulator
    with open("data/single_history_logs_old_2019_Jul_30_16_30.dat", "r") as rfile:
        history_logs_list_regulator = [eval(k) for k in rfile]

    # New data with government aid
    with open("data/single_history_logs_old_2019_Jul_31_11_37.dat", "r") as rfile:
        history_logs_list_aid = [eval(k) for k in rfile]

    # Data with no conditions
    with open("data/single_history_logs_old_2019_Jul_31_17_42.dat", "r") as rfile:
        history_logs_list_no_change = [eval(k) for k in rfile]

    new_data = history_logs_list_aid[0]
    old_data = history_logs_list_regulator[0]
    no_change_data = history_logs_list_no_change[0]

    for key in new_data.keys():
        if "firms_cash" in key or key == "market_diffvar" or "event" in key or "riskmodels" in key:
            pass
        elif key == "individual_contracts" or key == "reinsurance_contracts":
            time_range = min(len(old_data[key][0]), len(new_data[key][0]))
            avg_contract_per_firm_new = []
            avg_contract_per_firm_old = []
            for t in range(time_range):
                total_contracts = 0
                firm_counter = 0
                for i in range(len(new_data[key])):
                    if new_data[key][i][t] > 0:
                        total_contracts += new_data[key][i][t]
                        firm_counter += 1
                if firm_counter > 0:
                    avg_contract_per_firm_new.append(total_contracts/firm_counter)
                else:
                    avg_contract_per_firm_new.append(0)

                total_contracts = 0
                firm_counter = 0
                for i in range(len(old_data[key])):
                    if old_data[key][i][t] > 0:
                        total_contracts += old_data[key][i][t]
                        firm_counter += 1
                if firm_counter > 0:
                    avg_contract_per_firm_old.append(total_contracts / firm_counter)
                else:
                    avg_contract_per_firm_old.append(0)
            plt.plot(range(time_range), avg_contract_per_firm_new, label="New contract data", color="orange")
            plt.plot(range(time_range), avg_contract_per_firm_old, label="Old contract data", color="blue")
            plt.axhline(np.mean(avg_contract_per_firm_new), linestyle="--", label="Avg New Contract Data", color="orange")
            plt.axhline(np.mean(avg_contract_per_firm_old), linestyle="--", label="Avg Old Contract Data", color="blue")
            plt.legend()
            plt.suptitle(key)
            plt.xlabel("Time")
            plt.show()
        else:
            old_values = old_data[key]
            mean_old_values = np.mean(old_values)
            new_values = new_data[key]
            mean_new_values = np.mean(new_values)
            xvalues = np.arange(1000)
            plt.plot(xvalues, old_values, label='Regulator used (Old)', color="blue")
            plt.plot(xvalues, new_values, label='Aid used(New)', color="red")
            if "cum" not in key:
                plt.axhline(mean_new_values, linestyle='--', label="New Mean",  color="red")
                plt.axhline(mean_old_values, linestyle='--', label="Old Mean",  color="blue")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel(key)
            # plt.show()


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
                if "firms_cash" in key or key == "market_diffvar" or "event" in key or "riskmodels" in key:
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
                else:
                    if key not in avg_dict.keys():
                        avg_dict[key] = data[key]
                    else:
                        avg_dict[key] = [list1 + list2 for list1, list2 in zip(avg_dict[key], data[key])]
        for key in avg_dict.keys():
            avg_dict[key] = [value/len(data_dict) for value in avg_dict[key]]

    def plot(self):
        """Method to plot same type of data for different files on a plot.
        No accepted values.
        No return values."""
        for key in self.original_averages.keys():
            plt.figure()
            original_values = self.original_averages[key]
            mean_original_values = np.mean(original_values)
            new_values = self.new_averages[key]
            mean_new_values = np.mean(new_values)
            xvalues = np.arange(1000)
            plt.plot(xvalues[:500], original_values, label='Original Values', color="blue")
            plt.plot(xvalues, new_values, label='New Values', color="red")
            if self.extra:
                extra_values = self.extra_averages[key]
                mean_extra_values = np.mean(extra_values)
                plt.plot(xvalues, extra_values, label="Extra Values", color="yellow")
            if "cum" not in key:
                plt.axhline(mean_new_values, linestyle='--', label="New Mean",  color="red")
                plt.axhline(mean_original_values, linestyle='--', label="Original Mean",  color="blue")
                if self.extra:
                    plt.axhline(mean_extra_values, linestyle='--', label='Extra Mean', color='yellow')
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel(key)
        plt.show()

    def test(self):
        """Method to perform ks test on two sets of file data. Returns the D statistic and p-value.
            No accepted values.
            No return values."""
        for key in self.original_averages.keys():
            original_values = self.original_averages[key]
            new_values = self.new_averages[key][:500]
            D, p = ss.ks_2samp(original_values, new_values)
            print("%s has p value: %f and D: %f" % (key, p, D))


CD = CompareData("data/single_history_logs.dat", "data/single_history_logs_old_2019_Jul_30_16_30.dat", "data/single_history_logs_old_2019_Jul_31_11_37.dat")
CD.test()
CD.plot()
