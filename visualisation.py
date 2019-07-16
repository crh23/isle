# file to visualise data from a single and ensemble runs
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TimeSeries(object):
    def __init__(
        self,
        series_list,
        event_schedule,
        damage_schedule,
        title="",
        xlabel="Time",
        colour="k",
        axlst=None,
        fig=None,
        percentiles=None,
        alpha=0.7,
    ):
        self.series_list = series_list
        self.size = len(series_list)
        self.xlabel = xlabel
        self.colour = colour
        self.alpha = alpha
        self.percentiles = percentiles
        self.title = title
        self.timesteps = [
            t for t in range(len(series_list[0][0]))
        ]  # assume all data series are the same size
        self.events_schedule = event_schedule
        self.damage_schedule = damage_schedule
        if axlst is not None and fig is not None:
            self.axlst = axlst
            self.fig = fig
        else:
            self.fig, self.axlst = plt.subplots(self.size, sharex=True)

    def plot(self, schedule=False):
        event_categ_colours = ["r", "b", "g", "fuchsia"]
        for i, (series, series_label, fill_lower, fill_upper) in enumerate(
            self.series_list
        ):
            self.axlst[i].plot(self.timesteps, series, color=self.colour)
            self.axlst[i].set_ylabel(series_label)

            if fill_lower is not None and fill_upper is not None:
                self.axlst[i].fill_between(
                    self.timesteps,
                    fill_lower,
                    fill_upper,
                    color=self.colour,
                    alpha=self.alpha,
                )

            if schedule:  # Plots vertical lines for events if set.
                for categ in range(len(self.events_schedule)):
                    for event_time in self.events_schedule[categ]:
                        index = self.events_schedule[categ].index(event_time)
                        if (
                            self.damage_schedule[categ][index] > 0.5
                        ):  # Only plots line if event is significant
                            self.axlst[i].axvline(
                                event_time,
                                color=event_categ_colours[categ],
                                alpha=self.damage_schedule[categ][index],
                            )
        self.axlst[self.size - 1].set_xlabel(self.xlabel)
        self.fig.suptitle(self.title)

        return self.fig, self.axlst

    def save(self, filename):
        self.fig.savefig("{filename}".format(filename=filename))
        return


class InsuranceFirmAnimation(object):
    """Initialising method for the animation of insurance firm data.
        Accepts:
            cash_data: Type List of List of Lists: Contains the operational, ID and cash for each firm for each time.
            insure_contracts: Type List of Lists. Contains number of underwritten contracts for each firm for each time.
            event_schedule: Type List of Lists. Contains event times by category.
            type: Type String. Used to specify which file to save to.
            save: Type Boolean
            perils: Type Boolean. For if screen should flash during peril time.
        No return values.
    This class takes the cash and contract data of each firm over all time and produces an animation showing how the
    proportion of each for all operational firms changes with time. Allows it to be saved as an MP4."""

    def __init__(
        self,
        cash_data,
        insure_contracts,
        event_schedule,
        type,
        save=False,
        perils=False,
    ):
        # Converts list of events by category into list of all events.
        self.perils_condition = perils
        self.all_event_times = []
        for categ in event_schedule:
            self.all_event_times += categ
        self.all_event_times.sort()

        # Setting data and creating pie chart animations.
        self.cash_data = cash_data
        self.insurance_contracts = insure_contracts
        self.event_times_per_categ = event_schedule

        # If animation is saved or not
        self.save_condition = save
        self.type = type

    def animate(self):
        self.pies = [0, 0]
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.stream = self.data_stream()
        self.animate = animation.FuncAnimation(
            self.fig, self.update, repeat=False, interval=20, save_count=98
        )
        if self.save_condition:
            self.save()

    def data_stream(self):
        """Method to get the next set of firm data.
            No accpeted values
            Yields:
                firm_cash_list: Type List. Contains the cash of each firm.
                firm_id_list: Type List. Contains the unique ID of each firm.
                firm_contract_list: Type List. Contains the number of underwritten contracts for each firm.
        This iterates once every time it is called from the update method as it gets the next frame of data for the pie
        charts."""
        t = 0
        for timestep in self.cash_data:
            firm_cash_list = []
            firm_id_list = []
            firm_contract_list = []
            for (cash, id, operational) in timestep:
                if operational:
                    firm_id_list.append(id)
                    firm_cash_list.append(cash)
                    firm_contract_list.append(self.insurance_contracts[id][t])
            yield firm_cash_list, firm_id_list, firm_contract_list
            t += 1

    def update(self, i):
        """Method to update the animation frame.
            Accepts:
                i: Type Integer, iteration number.
            Returns:
                self.pies: Type List.
        This method is called or each iteration of the FuncAnimation and clears and redraws the pie charts onto the
        axis, getting data from data_stream method. Can also be set such that the figure flashes red at an event time."""
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.axis("equal")
        self.ax2.axis("equal")
        cash_list, id_list, con_list = next(self.stream)
        self.pies[0] = self.ax1.pie(cash_list, labels=id_list, autopct="%1.0f%%")
        self.ax1.set_title("Total cash : {:,.0f}".format(sum(cash_list)))
        self.pies[1] = self.ax2.pie(con_list, labels=id_list, autopct="%1.0f%%")
        self.ax2.set_title("Total contracts : {:,.0f}".format(sum(con_list)))
        self.fig.suptitle("%s  Timestep : %i" % (self.type, i))
        if self.perils_condition:
            if i == self.all_event_times[0]:
                self.fig.set_facecolor("r")
                self.all_event_times = self.all_event_times[1:]
            else:
                self.fig.set_facecolor("w")
        return self.pies

    def save(self):
        """Method to save the animation as mp4. Dependant on type of firm.
            No accepted values.
            No return values."""
        if self.type == "Insurance Firm":
            self.animate.save(
                "data/animated_insurfirm_pie.mp4", writer="ffmpeg", dpi=200, fps=10
            )
        elif self.type == "Reinsurance Firm":
            self.animate.save(
                "data/animated_reinsurefirm_pie.mp4", writer="ffmpeg", dpi=200, fps=10
            )
        else:
            print("Incorrect Type for Saving")


class visualisation(object):
    def __init__(self, history_logs_list):
        self.history_logs_list = history_logs_list
        return

    def insurer_pie_animation(self, run=0):
        data = self.history_logs_list[run]
        insurance_cash = np.array(data["insurance_firms_cash"])
        contract_data = self.history_logs_list[0]["individual_contracts"]
        event_schedule = self.history_logs_list[0]["rc_event_schedule_initial"]
        self.ins_pie_anim = InsuranceFirmAnimation(
            insurance_cash, contract_data, event_schedule, "Insurance Firm", save=True
        )
        self.ins_pie_anim.animate()
        return self.ins_pie_anim

    def reinsurer_pie_animation(self, run=0):
        data = self.history_logs_list[run]
        reinsurance_cash = np.array(data["reinsurance_firms_cash"])
        contract_data = self.history_logs_list[0]["reinsurance_contracts"]
        event_schedule = self.history_logs_list[0]["rc_event_schedule_initial"]
        self.reins_pie_anim = InsuranceFirmAnimation(
            reinsurance_cash,
            contract_data,
            event_schedule,
            "Reinsurance Firm",
            save=True,
        )
        self.reins_pie_anim.animate()
        return self.reins_pie_anim

    def insurer_time_series(
        self,
        runs=None,
        axlst=None,
        fig=None,
        title="Insurer",
        colour="black",
        percentiles=[25, 75],
    ):
        # runs should be a list of the indexes you want included in the ensemble for consideration
        if runs:
            data = [self.history_logs_list[x] for x in runs]
        else:
            data = self.history_logs_list

        # Take the element-wise means/medians of the ensemble set (axis=0)
        contracts_agg = [
            history_logs["total_contracts"] for history_logs in self.history_logs_list
        ]
        profitslosses_agg = [
            history_logs["total_profitslosses"]
            for history_logs in self.history_logs_list
        ]
        operational_agg = [
            history_logs["total_operational"] for history_logs in self.history_logs_list
        ]
        cash_agg = [
            history_logs["total_cash"] for history_logs in self.history_logs_list
        ]
        premium_agg = [
            history_logs["market_premium"] for history_logs in self.history_logs_list
        ]

        contracts = np.mean(contracts_agg, axis=0)
        profitslosses = np.mean(profitslosses_agg, axis=0)
        operational = np.median(operational_agg, axis=0)
        cash = np.median(cash_agg, axis=0)
        premium = np.median(premium_agg, axis=0)

        events = self.history_logs_list[0]["rc_event_schedule_initial"]
        damages = self.history_logs_list[0]["rc_event_damage_initial"]

        self.ins_time_series = TimeSeries(
            [
                (
                    contracts,
                    "Contracts",
                    np.percentile(contracts_agg, percentiles[0], axis=0),
                    np.percentile(contracts_agg, percentiles[1], axis=0),
                ),
                (
                    profitslosses,
                    "Profitslosses",
                    np.percentile(profitslosses_agg, percentiles[0], axis=0),
                    np.percentile(profitslosses_agg, percentiles[1], axis=0),
                ),
                (
                    operational,
                    "Operational",
                    np.percentile(operational_agg, percentiles[0], axis=0),
                    np.percentile(operational_agg, percentiles[1], axis=0),
                ),
                (
                    cash,
                    "Cash",
                    np.percentile(cash_agg, percentiles[0], axis=0),
                    np.percentile(cash_agg, percentiles[1], axis=0),
                ),
                (
                    premium,
                    "Premium",
                    np.percentile(premium_agg, percentiles[0], axis=0),
                    np.percentile(premium_agg, percentiles[1], axis=0),
                ),
            ],
            events,
            damages,
            title=title,
            xlabel="Time",
            axlst=axlst,
            fig=fig,
            colour=colour,
        )
        self.ins_time_series.plot(schedule=True)
        return self.ins_time_series

    def reinsurer_time_series(
        self,
        runs=None,
        axlst=None,
        fig=None,
        title="Reinsurer",
        colour="black",
        percentiles=[25, 75],
    ):
        # runs should be a list of the indexes you want included in the ensemble for consideration
        if runs:
            data = [self.history_logs_list[x] for x in runs]
        else:
            data = self.history_logs_list

        # Take the element-wise means/medians of the ensemble set (axis=0)
        reincontracts_agg = [
            history_logs["total_reincontracts"]
            for history_logs in self.history_logs_list
        ]
        reinprofitslosses_agg = [
            history_logs["total_reinprofitslosses"]
            for history_logs in self.history_logs_list
        ]
        reinoperational_agg = [
            history_logs["total_reinoperational"]
            for history_logs in self.history_logs_list
        ]
        reincash_agg = [
            history_logs["total_reincash"] for history_logs in self.history_logs_list
        ]
        catbonds_number_agg = [
            history_logs["total_catbondsoperational"]
            for history_logs in self.history_logs_list
        ]

        reincontracts = np.mean(reincontracts_agg, axis=0)
        reinprofitslosses = np.mean(reinprofitslosses_agg, axis=0)
        reinoperational = np.median(reinoperational_agg, axis=0)
        reincash = np.median(reincash_agg, axis=0)
        catbonds_number = np.median(catbonds_number_agg, axis=0)

        events = self.history_logs_list[0]["rc_event_schedule_initial"]
        damages = self.history_logs_list[0]["rc_event_damage_initial"]

        self.reins_time_series = TimeSeries(
            [
                (
                    reincontracts,
                    "Contracts",
                    np.percentile(reincontracts_agg, percentiles[0], axis=0),
                    np.percentile(reincontracts_agg, percentiles[1], axis=0),
                ),
                (
                    reinprofitslosses,
                    "Profitslosses",
                    np.percentile(reinprofitslosses_agg, percentiles[0], axis=0),
                    np.percentile(reinprofitslosses_agg, percentiles[1], axis=0),
                ),
                (
                    reinoperational,
                    "Operational",
                    np.percentile(reinoperational_agg, percentiles[0], axis=0),
                    np.percentile(reinoperational_agg, percentiles[1], axis=0),
                ),
                (
                    reincash,
                    "Cash",
                    np.percentile(reincash_agg, percentiles[0], axis=0),
                    np.percentile(reincash_agg, percentiles[1], axis=0),
                ),
                (
                    catbonds_number,
                    "Activate Cat Bonds",
                    np.percentile(catbonds_number_agg, percentiles[0], axis=0),
                    np.percentile(catbonds_number_agg, percentiles[1], axis=0),
                ),
            ],
            events,
            damages,
            title=title,
            xlabel="Time",
            axlst=axlst,
            fig=fig,
            colour=colour,
        )
        self.reins_time_series.plot()
        return self.reins_time_series

    def metaplotter_timescale(self):
        # Take the element-wise means/medians of the ensemble set (axis=0)
        contracts = np.mean(
            [
                history_logs["total_contracts"]
                for history_logs in self.history_logs_list
            ],
            axis=0,
        )
        profitslosses = np.mean(
            [
                history_logs["total_profitslosses"]
                for history_logs in self.history_logs_list
            ],
            axis=0,
        )
        operational = np.median(
            [
                history_logs["total_operational"]
                for history_logs in self.history_logs_list
            ],
            axis=0,
        )
        cash = np.median(
            [history_logs["total_cash"] for history_logs in self.history_logs_list],
            axis=0,
        )
        premium = np.median(
            [history_logs["market_premium"] for history_logs in self.history_logs_list],
            axis=0,
        )
        reincontracts = np.mean(
            [
                history_logs["total_reincontracts"]
                for history_logs in self.history_logs_list
            ],
            axis=0,
        )
        reinprofitslosses = np.mean(
            [
                history_logs["total_reinprofitslosses"]
                for history_logs in self.history_logs_list
            ],
            axis=0,
        )
        reinoperational = np.median(
            [
                history_logs["total_reinoperational"]
                for history_logs in self.history_logs_list
            ],
            axis=0,
        )
        reincash = np.median(
            [history_logs["total_reincash"] for history_logs in self.history_logs_list],
            axis=0,
        )
        catbonds_number = np.median(
            [
                history_logs["total_catbondsoperational"]
                for history_logs in self.history_logs_list
            ],
            axis=0,
        )
        return

    def show(self):
        plt.show()
        return


class compare_riskmodels(object):
    def __init__(self, vis_list, colour_list):
        # take in list of visualisation objects and call their plot methods
        self.vis_list = vis_list
        self.colour_list = colour_list

    def create_insurer_timeseries(self, fig=None, axlst=None, percentiles=[25, 75]):
        # create the time series for each object in turn and superpose them?
        fig = axlst = None
        for vis, colour in zip(self.vis_list, self.colour_list):
            (fig, axlst) = vis.insurer_time_series(
                fig=fig, axlst=axlst, colour=colour, percentiles=percentiles
            )

    def create_reinsurer_timeseries(self, fig=None, axlst=None, percentiles=[25, 75]):
        # create the time series for each object in turn and superpose them?
        fig = axlst = None
        for vis, colour in zip(self.vis_list, self.colour_list):
            (fig, axlst) = vis.reinsurer_time_series(
                fig=fig, axlst=axlst, colour=colour, percentiles=percentiles
            )

    def show(self):
        plt.show()

    def save(self):
        # logic to save plots
        pass


if __name__ == "__main__":
    # use argparse to handle command line arguments
    parser = argparse.ArgumentParser(description="Model the Insurance sector")
    parser.add_argument(
        "--single",
        action="store_true",
        help="plot time series of a single run of the insurance model",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="plot the result of an ensemble of replicatons of the insurance model",
    )

    args = parser.parse_args()
    args.single = True
    if args.single:

        # load in data from the history_logs dictionarywith open("data/history_logs.dat","r") as rfile:
        with open("data/history_logs.dat", "r") as rfile:
            history_logs_list = [eval(k) for k in rfile]  # one dict on each line

        # first create visualisation object, then create graph/animation objects as necessary
        vis = visualisation(history_logs_list)
        vis.insurer_pie_animation()
        vis.reinsurer_pie_animation()
        # vis.insurer_time_series()
        # vis.reinsurer_time_series()
        vis.show()
        N = len(history_logs_list)

    if args.comparison:

        # for each run, generate an animation and time series for insurer and reinsurer
        # TODO: provide some way for these to be lined up nicely rather than having to manually arrange screen
        # for i in range(N):
        #    vis.insurer_pie_animation(run=i)
        #    vis.insurer_time_series(runs=[i])
        #    vis.reinsurer_pie_animation(run=i)
        #    vis.reinsurer_time_series(runs=[i])
        #    vis.show()
        vis_list = []
        filenames = [
            "./data/" + x + "_history_logs.dat" for x in ["one", "two", "three", "four"]
        ]
        for filename in filenames:
            with open(filename, "r") as rfile:
                history_logs_list = [eval(k) for k in rfile]  # one dict on each line
                vis_list.append(visualisation(history_logs_list))

        colour_list = ["blue", "yellow", "red", "green"]
        cmp_rsk = compare_riskmodels(vis_list, colour_list)
        cmp_rsk.create_insurer_timeseries(percentiles=[10, 90])
        cmp_rsk.create_reinsurer_timeseries(percentiles=[10, 90])
        cmp_rsk.show()
