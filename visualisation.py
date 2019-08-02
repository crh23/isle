# file to visualise data from a single and ensemble runs
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import isleconfig
import pickle
import scipy
import scipy.stats
from matplotlib.offsetbox import AnchoredText
import time
import os

if not os.path.isdir("figures"):
    os.makedirs("figures")


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
        """Intialisation method for creating timeseries.
            Accepts:
                series_list: Type List. Contains contract, cash, operational, premium, profitloss data.
                event_schedule: Type List of Lists. Used to plot event times on timeseries if a single run.
                damage_schedule: Type List of Lisst. Used for plotting event times on timeseries if single run.
                title: Type string.
                xlabel: Type string.
                colour: Type string.
                axlist: Type None or list of axes for subplots.
                fig: Type None or figure.
                percentiles: Type list. Has the percentiles within which data is plotted for.
                alpha: Type Integer. Alpha of graph plots.
            No return values"""
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

    def plot(self):
        """Method  to plot time series.
            No accepted values.
            Returns:
                self.fig: Type figure. Used to for saving graph to a file.
                self.axlst: Type axes list.
        This method is called to plot a timeseries for five subplots of data for insurers/reinsurers. If called for a
        single run event times are plotted as vertical lines, if an ensemble run then no events but the average data is
        plotted with percentiles as deviations to the average."""
        single_categ_colours = ["b", "b", "b", "b"]
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

            if (
                self.events_schedule is not None
            ):  # Plots vertical lines for events if set.
                for categ in range(len(self.events_schedule)):
                    for event_time in self.events_schedule[categ]:
                        index = self.events_schedule[categ].index(event_time)
                        if (
                            self.damage_schedule[categ][index] > 0.5
                        ):  # Only plots line if event is significant
                            self.axlst[i].axvline(
                                event_time,
                                color=single_categ_colours[categ],
                                alpha=self.damage_schedule[categ][index],
                            )
        self.axlst[self.size - 1].set_xlabel(self.xlabel)
        self.fig.suptitle(self.title)

        return self.fig, self.axlst


class InsuranceFirmAnimation(object):
    def __init__(
        self, cash_data, insure_contracts, event_schedule, type, save=True, perils=True
    ):
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
        """Method to call animation of pie charts.
            No accepted values.
            No returned values.
        This method is called after the simulation class is initialised to start the animation of pie charts, and will
        save it as an mp4 if applicable."""
        self.pies = [0, 0]
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.stream = self.data_stream()
        self.animate = animation.FuncAnimation(
            self.fig, self.update, repeat=False, interval=20, save_count=998
        )
        if self.save_condition:
            self.save()

    def data_stream(self):
        """Method to get the next set of firm data.
            No accepted values
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
                self.fig.suptitle("EVENT AT TIME %i!" % i)
                self.all_event_times = self.all_event_times[1:]
        return self.pies

    def save(self):
        """Method to save the animation as mp4. Dependant on type of firm.
            No accepted values.
            No return values."""
        if self.type == "Insurance Firm":
            self.animate.save(
                "figures/animated_insurfirm_pie.mp4", writer="ffmpeg", dpi=200, fps=10
            )
        elif self.type == "Reinsurance Firm":
            self.animate.save(
                "figures/animated_reinsurefirm_pie.mp4",
                writer="ffmpeg",
                dpi=200,
                fps=10,
            )
        else:
            print("Incorrect Type for Saving")


class visualisation(object):
    def __init__(self, history_logs_list):
        """Initialises visualisation class for all data.
            Accepts:
                history_logs_list: Type List of DataDicts. Each element is a different replication/run. Each DataDict
                                    contains all info for that run.
            No return values."""
        self.history_logs_list = history_logs_list
        self.scatter_data = {}

    def insurer_pie_animation(self, run=0):
        """Method to created animated pie chart of cash and contract proportional per operational insurance firm.
            Accepts:
                run: Type Integer. Which replication/run is wanted. Allows loops or specific runs.
            Returns:
                self.ins_pie_anim: Type animation class instance. Not used outside this method but is saved."""
        data = self.history_logs_list[run]
        insurance_cash = np.array(data["insurance_firms_cash"])
        contract_data = data["individual_contracts"]
        event_schedule = data["rc_event_schedule_initial"]
        self.ins_pie_anim = InsuranceFirmAnimation(
            insurance_cash, contract_data, event_schedule, "Insurance Firm", save=True
        )
        self.ins_pie_anim.animate()
        return self.ins_pie_anim

    def reinsurer_pie_animation(self, run=0):
        """Method to created animated pie chart of cash and contract proportional per operational reinsurance firm.
            Accepts:
                run: Type Integer. Which replication/run is wanted. Allows loops or specific runs.
            Returns:
                self.reins_pie_anim: Type animation class instance. Not used outside this method but is saved."""
        data = self.history_logs_list[run]
        reinsurance_cash = np.array(data["reinsurance_firms_cash"])
        contract_data = data["reinsurance_contracts"]
        event_schedule = data["rc_event_schedule_initial"]
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
        singlerun=True,
        axlst=None,
        fig=None,
        title="Insurer",
        colour="black",
        percentiles=[25, 75],
    ):
        """Method to create a timeseries for insurance firms' data.
            Accepts:
                singlerun: Type Boolean. Sets event schedule if a single run.
                axlst: Type axes list, normally None and created later.
                fig: Type figure, normally None and created later.
                title: Type String.
                colour: Type String.
                percentiles: Type List. For ensemble runs to plot outer limits of data.
            Returns:
                fig: Type figure. Used to save times series.
                axlst: Type axes list. Not used.
        This method is called to plot a timeseries of contract, cash, operational, profitloss, and premium data for
        insurance firms from saved data. Also sets event schedule for single run data, to plots event times on
        timeseries, as this is only helpful in this case."""
        if singlerun:
            events = self.history_logs_list[0]["rc_event_schedule_initial"]
            damages = self.history_logs_list[0]["rc_event_damage_initial"]
        else:
            events = None
            damages = None

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
        fig, axlst = self.ins_time_series.plot()
        return fig, axlst

    def reinsurer_time_series(
        self,
        singlerun=True,
        axlst=None,
        fig=None,
        title="Reinsurer",
        colour="black",
        percentiles=[25, 75],
    ):
        """Method to create a timeseries for reinsurance firms' data.
            Accepts:
                singlerun: Type Boolean. Sets event schedule if a single run.
                axlst: Type axes list, normally None and created later.
                fig: Type figure, normally None and created later.
                title: Type String.
                colour: Type String.
                percentiles: Type List. For ensemble runs to plot outer limits of data.
            Returns:
                fig: Type figure. Used to save times series.
                axlst: Type axes list. Not used.
        This method is called to plot a timeseries of contract, cash, operational, profitloss, and catbond data for
        reinsurance firms from saved data. Also sets event schedule for single run data, to plots event times on
        timeseries, as this is only helpful in this case."""
        if singlerun:
            events = self.history_logs_list[0]["rc_event_schedule_initial"]
            damages = self.history_logs_list[0]["rc_event_damage_initial"]
        else:
            events = None
            damages = None

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
        fig, axlst = self.reins_time_series.plot()
        return fig, axlst

    def aux_clustered_exit_records(self, exits):
        """Auxiliary method for creation of data series on clustered events such as firm market exits.
        Will take an unclustered series and aggregate every series of non-zero elements into
        the first element of that series.
            Arguments:
                exits: numpy ndarray or list  - unclustered series
            Returns:
                numpy ndarray of the same length as argument "exits": the clustered series."""
        exits2 = []
        ci = False
        cidx = 0
        for ee in exits:
            if ci:
                exits2.append(0)
                if ee > 0:
                    exits2[cidx] += ee
                else:
                    ci = False
            else:
                exits2.append(ee)
                if ee > 0:
                    ci = True
                    cidx = len(exits2) - 1

        return np.asarray(exits2, dtype=np.float64)

    def populate_scatter_data(self):
        """Method to generate data samples that do not have a time component (e.g. the size of bankruptcy events, i.e.
                how many firms exited each time.
                The method saves these in the instance variable self.scatter_data. This variable is of type dict.
            Arguments: None.
            Returns: None."""

        """Record data on sizes of unrecovered_claims"""
        self.scatter_data["unrecovered_claims"] = []
        for hlog in self.history_logs_list:  # for each replication
            urc = np.diff(np.asarray(hlog["cumulative_unrecovered_claims"]))
            self.scatter_data["unrecovered_claims"] = np.hstack(
                [self.scatter_data["unrecovered_claims"], np.extract(urc > 0, urc)]
            )

        """Record data on sizes of unrecovered_claims"""
        self.scatter_data["relative_unrecovered_claims"] = []
        for hlog in self.history_logs_list:  # for each replication
            urc = np.diff(np.asarray(hlog["cumulative_unrecovered_claims"]))
            tcl = np.diff(np.asarray(hlog["cumulative_claims"]))
            rurc = urc / tcl
            self.scatter_data["relative_unrecovered_claims"] = np.hstack(
                [self.scatter_data["unrecovered_claims"], np.extract(rurc > 0, rurc)]
            )
            try:
                assert (
                    np.isinf(self.scatter_data["relative_unrecovered_claims"]).any()
                    == False
                )
            except:
                pass
                # pdb.set_trace()

        """Record data on sizes of bankruptcy_events"""
        self.scatter_data["bankruptcy_events"] = []
        self.scatter_data["bankruptcy_events_relative"] = []
        self.scatter_data["bankruptcy_events_clustered"] = []
        self.scatter_data["bankruptcy_events_relative_clustered"] = []
        for hlog in self.history_logs_list:  # for each replication
            """Obtain numbers of operational firms. This is for computing the relative share of exiting firms."""
            in_op = np.asarray(hlog["total_operational"])[:-1]
            rein_op = np.asarray(hlog["total_reinoperational"])[:-1]
            op = in_op + rein_op
            exits = np.diff(
                np.asarray(hlog["cumulative_market_exits"], dtype=np.float64)
            )
            assert (exits <= op).all()
            op[op == 0] = 1

            """Obtain exits and relative exits"""
            # exits = np.diff(np.asarray(hlog["cumulative_market_exits"], dtype=np.float64)) # used above already
            rel_exits = exits / op

            """Obtain clustered exits (absolute and relative)"""
            exits2 = self.aux_clustered_exit_records(exits)
            rel_exits2 = exits2 / op

            """Record data"""
            self.scatter_data["bankruptcy_events"] = np.hstack(
                [self.scatter_data["bankruptcy_events"], np.extract(exits > 0, exits)]
            )
            self.scatter_data["bankruptcy_events_relative"] = np.hstack(
                [
                    self.scatter_data["bankruptcy_events_relative"],
                    np.extract(rel_exits > 0, rel_exits),
                ]
            )
            self.scatter_data["bankruptcy_events_clustered"] = np.hstack(
                [
                    self.scatter_data["bankruptcy_events_clustered"],
                    np.extract(exits2 > 0, exits2),
                ]
            )
            self.scatter_data["bankruptcy_events_relative_clustered"] = np.hstack(
                [
                    self.scatter_data["bankruptcy_events_relative_clustered"],
                    np.extract(rel_exits2 > 0, rel_exits2),
                ]
            )

    def show(self):
        plt.show()
        return


class compare_riskmodels(object):
    def __init__(self, vis_list, colour_list):
        """Initialises compare_riskmodels class.
            Accepts:
                vis_list: Type List of Visualisation class instances. Each instance is a different no. of risk models.
                colour_list: Type List of string(colours).
        Takes in list of visualisation objects and call their plot methods."""
        self.vis_list = vis_list
        self.colour_list = colour_list
        self.insurer_fig = self.insurer_axlst = None
        self.reinsurer_fig = self.reinsurer_axlst = None

    def create_insurer_timeseries(self, fig=None, axlst=None, percentiles=[25, 75]):
        """Method to create separate insurer time series for all numbers of risk models using visualisations'
        insurer_time_series method. Loops through each separately and they are then saved automatically. Used for
        ensemble runs.
            Accepts:
                fig: Type figure.
                axlst: Type axes list.
                percentiles: Type List.
            No return values."""
        risk_model = 0
        for vis, colour in zip(self.vis_list, self.colour_list):
            risk_model += 1
            (self.insurer_fig, self.insurer_axlst) = vis.insurer_time_series(
                singlerun=False,
                fig=fig,
                axlst=axlst,
                colour=colour,
                percentiles=percentiles,
                title="%i Risk Model Insurer" % risk_model,
            )
            self.insurer_fig.savefig(
                "figures/"
                + str(risk_model)
                + "risk_model(s)_insurer_ensemble_timeseries.png"
            )
            print(
                "Saved "
                + str(risk_model)
                + " risk_model(s)_insurer_ensemble_timeseries"
            )

    def create_reinsurer_timeseries(self, fig=None, axlst=None, percentiles=[25, 75]):
        """Method to create separate reinsurer time series for all numbers of risk models using visualisations'
        reinsurer_time_series method. Loops through each separately and they are then saved automatically. Used for
        ensemble runs.
            Accepts:
                fig: Type figure.
                axlst: Type axes list.
                percentiles: Type List.
            No return values."""
        risk_model = 0
        for vis, colour in zip(self.vis_list, self.colour_list):
            risk_model += 1
            (self.reinsurer_fig, self.reinsurer_axlst) = vis.reinsurer_time_series(
                singlerun=False,
                fig=fig,
                axlst=axlst,
                colour=colour,
                percentiles=percentiles,
                title="%i Risk Model Reinsurer" % risk_model,
            )
            self.reinsurer_fig.savefig(
                "figures/"
                + str(risk_model)
                + "risk_model(s)_reinsurer_ensemble_timeseries.png"
            )
            print(
                "Saved "
                + str(risk_model)
                + " risk_model(s)_reinsurer_ensemble_timeseries"
            )

    def show(self):
        plt.show()


class CDF_distribution_plot:
    """Class for CDF/cCDF distribution plots using class CDFDistribution. This class arranges as many such plots stacked
    in one diagram as there are series in the history logs they are created from, i.e. len(vis_list)."""

    def __init__(
        self,
        vis_list,
        colour_list,
        quantiles=[0.25, 0.75],
        variable="reinsurance_firms_cash",
        timestep=-1,
        plot_cCDF=True,
    ):
        """Constructor.
            Arguments:
                vis_list: list of visualisation objects - objects hilding the data
                colour list: list of str                - colors to be used for each plot
                quantiles: list of float of length 2    - lower and upper quantile for inter quantile range in plot
                variable: string (must be a valid dict key in vis_list[i].history_logs_list
                                                        - the history log variable for which the distribution is plotted
                                                            (will be either "insurance_firms_cash" or "reinsurance_firms_cash")
                timestep: int                           - timestep at which the distribution to be plotted is taken
                plot_cCDF: bool                         - plot survival function (cCDF) instead of CDF
            Returns class instance."""
        self.vis_list = vis_list
        self.colour_list = colour_list
        self.lower_quantile, self.upper_quantile = quantiles
        self.variable = variable
        self.timestep = timestep

    def generate_plot(self, xlabel=None, filename=None):
        """Method to generate and save the plot.
            Arguments:
                xlabel: str or None     - the x axis label
                filename: str or None   - the filename without ending
            Returns None.
        This method unpacks the variable wanted from the history log data then uses the CDFDistribution class to plot it"""

        """Set x axis label and filename to default if not provided"""
        xlabel = xlabel if xlabel is not None else self.variable
        filename = (
            filename if filename is not None else "figures/CDF_plot_" + self.variable
        )

        """Create figure with correct number of subplots"""
        self.fig, self.ax = plt.subplots(nrows=len(self.vis_list))

        """find max and min values"""
        """combine all data sets"""
        all_data = np.asarray([])
        for i in range(len(self.vis_list)):
            """Extract firm records from history logs"""
            series_x = [
                replication[self.variable][self.timestep]
                for replication in self.vis_list[i].history_logs_list
            ]
            """Extract the capital holdings from the tuple"""
            for j in range(len(series_x)):
                series_x[j] = [firm[0] for firm in series_x[j] if firm[2]]
            series_x = np.hstack(series_x)
            all_data = np.hstack([all_data, series_x])
        """Catch empty data sets"""
        if len(all_data) == 0:
            return
        minmax = (np.min(all_data), np.max(all_data) / 2.0)

        """Loop through simulation record series, populate subplot by subplot"""
        for i in range(len(self.vis_list)):
            """Extract firm records from history logs"""
            series_x = [
                replication[self.variable][self.timestep]
                for replication in self.vis_list[i].history_logs_list
            ]
            """Extract the capital holdings from the tuple"""
            for j in range(len(series_x)):
                series_x[j] = [firm[0] for firm in series_x[j] if firm[2]]
            """Create CDFDistribution object and populate the subfigure using it"""
            VDP = CDFDistribution(series_x)
            c_xlabel = "" if i < len(self.vis_list) - 1 else xlabel
            VDP.plot(
                ax=self.ax[i],
                ylabel="cCDF " + str(i + 1) + "RM",
                xlabel=c_xlabel,
                upper_quantile=self.upper_quantile,
                lower_quantile=self.lower_quantile,
                color=self.colour_list[i],
                plot_cCDF=True,
                xlims=minmax,
            )

        """Finish and save figure"""
        self.fig.tight_layout()
        self.fig.savefig(filename + ".pdf")
        self.fig.savefig(filename + ".png", density=300)
        print("Saved " + self.variable + " CDF")


class Histogram_plot:
    """Class for Histogram plots using class Histograms. This class arranges as many such plots stacked in one diagram
    as there are series in the history logs they are created from, i.e. len(vis_list)."""

    def __init__(self, vis_list, colour_list, variable="bankruptcy_events"):
        """Constructor.
            Arguments:
                vis_list: list of visualisation objects - objects hilding the data
                colour list: list of str                - colors to be used for each plot
                variable: string (must be a valid dict key in vis_list[i].scatter_data
                                                        - the history log variable for which the distribution is plotted
            Returns class instance."""
        self.vis_list = vis_list
        self.colour_list = colour_list
        self.variable = variable

    def generate_plot(
        self, xlabel=None, filename=None, logscale=False, minmax=None, VaR005guess=0.3
    ):
        """Method to generate and save the plot.
            Arguments:
                xlabel: str or None     - the x axis label
                filename: str or None   - the filename without ending
            Returns None."""

        """Set x axis label and filename to default if not provided"""
        xlabel = xlabel if xlabel is not None else self.variable
        filename = (
            filename
            if filename is not None
            else "figures/Histogram_plot_" + self.variable
        )

        """Create figure with correct number of subplots"""
        self.fig, self.ax = plt.subplots(nrows=len(self.vis_list))

        """find max and min values"""
        """combine all data sets"""
        all_data = [np.asarray(vl.scatter_data[self.variable]) for vl in self.vis_list]
        with open("scatter_data.pkl", "wb") as wfile:
            pickle.dump(all_data, wfile)
        all_data = np.hstack(all_data)
        """Catch empty data sets"""
        if len(all_data) == 0:
            return
        if minmax is None:
            minmax = (np.min(all_data), np.max(all_data))
        num_bins = min(25, len(np.unique(all_data)))

        """Loop through simulation record series, populate subplot by subplot"""
        for i in range(len(self.vis_list)):
            """Extract records from history logs"""
            scatter_data = self.vis_list[i].scatter_data[self.variable]
            """Create Histogram object and populate the subfigure using it"""
            H = Histogram(scatter_data)
            c_xlabel = "" if i < len(self.vis_list) - 1 else xlabel
            c_xtralabel = (
                str(i + 1) + " risk models" if i > 0 else str(i + 1) + " risk model"
            )
            c_ylabel = "Frequency" if i == 2 else ""
            H.plot(
                ax=self.ax[i],
                ylabel=c_ylabel,
                xtralabel=c_xtralabel,
                xlabel=c_xlabel,
                color=self.colour_list[i],
                num_bins=num_bins,
                logscale=logscale,
                xlims=minmax,
            )
            VaR005 = sorted(scatter_data, reverse=True)[
                int(round(len(scatter_data) * 200.0 / 4000.0))
            ]
            realized_events_beyond = len(
                np.extract(scatter_data > VaR005guess, scatter_data)
            )
            realized_expected_shortfall = (
                np.mean(np.extract(scatter_data > VaR005guess, scatter_data))
                - VaR005guess
            )
            print(
                self.variable,
                c_xtralabel,
                "Slope: ",
                1 / scipy.stats.expon.fit(scatter_data)[0],
                "1/200 threshold: ",
                VaR005,
                " #Events beyond: ",
                realized_events_beyond,
                "Relative: ",
                realized_events_beyond * 1.0 / len(scatter_data),
                " Expected shortfall: ",
                realized_expected_shortfall,
            )

        """Finish and save figure"""
        self.fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        self.fig.savefig(filename + ".pdf")
        self.fig.savefig(filename + ".png", density=300)
        print("Saved " + self.variable + " histogram")


class CDFDistribution:
    def __init__(self, samples_x):
        """Constructor.
            Arguments:
                samples_x: list of list or ndarray of int or float - list of samples to be visualized.
            Returns:
                Class instance"""
        self.samples_x = []
        self.samples_y = []
        for x in samples_x:
            if len(x) > 0:
                x = np.sort(np.asarray(x, dtype=np.float64))
                y = (np.arange(len(x), dtype=np.float64) + 1) / len(x)
                self.samples_x.append(x)
                self.samples_y.append(y)
        self.series_y = None
        self.median_x = None
        self.mean_x = None
        self.quantile_series_x = None
        self.quantile_series_y_lower = None
        self.quantile_series_y_upper = None

    def make_figure(self, upper_quantile=0.25, lower_quantile=0.75):
        """Method to do the necessary computations to create the CDF plot (incl. mean, median, quantiles.
           This method populates the variables that are plotted.
            Arguments:
                upper_quantile: float \in [0,1] - upper quantile threshold
                lower_quantile: float \in [0,1] - lower quantile threshold
            Returns None."""

        """Obtain ordered set of all y values"""
        self.series_y = np.unique(np.sort(np.hstack(self.samples_y)))

        """Obtain x coordinates corresponding to the full ordered set of all y values (self.series_y) for each series"""
        set_of_series_x = []
        for i in range(len(self.samples_x)):
            x = [
                self.samples_x[i][np.argmax(self.samples_y[i] >= y)]
                if self.samples_y[i][0] <= y
                else 0
                for y in self.series_y
            ]
            set_of_series_x.append(x)

        """Join x coordinates to matrix of size m x n (n: number of series, m: length of ordered set of y values (self.series_y))"""
        series_matrix_x = np.vstack(set_of_series_x)

        """Compute x quantiles, median, mean across all series"""
        quantile_lower_x = np.quantile(series_matrix_x, 0.25, axis=0)
        quantile_upper_x = np.quantile(series_matrix_x, 0.75, axis=0)
        self.median_x = np.quantile(series_matrix_x, 0.50, axis=0)
        self.mean_x = series_matrix_x.mean(axis=0)

        """Obtain x coordinates for quantile plots. This is the ordered set of all x coordinates in lower and upper quantile series."""
        self.quantile_series_x = np.unique(
            np.sort(np.hstack([quantile_lower_x, quantile_upper_x]))
        )

        """Obtain y coordinates for quantile plots. This is one y value for each x coordinate."""
        # self.quantile_series_y_lower = [self.series_y[np.argmax(quantile_lower_x>=x)] if quantile_lower_x[0]<=x else 0 for x in self.quantile_series_x]
        self.quantile_series_y_lower = np.asarray(
            [
                self.series_y[np.argmax(quantile_lower_x >= x)]
                if np.sum(np.argmax(quantile_lower_x >= x)) > 0
                else np.max(self.series_y)
                for x in self.quantile_series_x
            ]
        )
        self.quantile_series_y_upper = np.asarray(
            [
                self.series_y[np.argmax(quantile_upper_x >= x)]
                if quantile_upper_x[0] <= x
                else 0
                for x in self.quantile_series_x
            ]
        )

        """The first value of lower must be zero"""
        self.quantile_series_y_lower[0] = 0.0

        print(list(self.median_x), "\n\n", list(self.series_y), "\n\n\n\n")

    def reverse_CDF(self):
        """Method to reverse the CDFs and obtain the complementary CDFs (survival functions) instead.
           The method overwrites the attributes used for plotting.
            Arguments: None.
            Returns: None."""
        self.series_y = 1.0 - self.series_y
        self.quantile_series_y_lower = 1.0 - self.quantile_series_y_lower
        self.quantile_series_y_upper = 1.0 - self.quantile_series_y_upper

    def plot(
        self,
        ax=None,
        ylabel="CDF(x)",
        xlabel="y",
        upper_quantile=0.25,
        lower_quantile=0.75,
        force_recomputation=False,
        show=False,
        outputname=None,
        color="C2",
        plot_cCDF=False,
        xlims=None,
    ):
        """Method to compile the plot. The plot is added to a provided matplotlib axes object or a new one is created.
            Arguments:
                ax: matplitlib axes             - the system of coordinates into which to plot
                ylabel: str                     - y axis label
                xlabel: str                     - x axis label
                upper_quantile: float \in [0,1] - upper quantile threshold
                lower_quantile: float \in [0,1] - lower quantile threshold
                force_recomputation: bool       - force re-computation of plots
                show: bool                      - show plot
                outputname: str                 - output file name without ending
                color: str or other admissible matplotlib color label - color to use for the plot
                plot_cCDF: bool                 - plot survival function (cCDF) instead of CDF
            Returns: None."""

        """If data set is empty, return without plotting"""
        if self.samples_x == []:
            return

        """Create figure if none was provided"""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        """Compute plots if not already done or if recomputation was requested"""
        if (self.series_y is None) or force_recomputation:
            self.make_figure(upper_quantile, lower_quantile)

        """Switch to cCDF if requested"""
        if plot_cCDF:
            self.reverse_CDF()

        """Plot"""
        ax.fill_between(
            self.quantile_series_x,
            self.quantile_series_y_lower,
            self.quantile_series_y_upper,
            facecolor=color,
            alpha=0.25,
        )
        ax.plot(self.median_x, self.series_y, color=color)
        ax.plot(self.mean_x, self.series_y, dashes=[3, 3], color=color)

        """Set plot attributes"""
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        """Set xlim if requested"""
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])

        """Save if filename provided"""
        if outputname is not None:
            plt.savefig(outputname + ".pdf")
            plt.savefig(outputname + ".png", density=300)

        """Show if requested"""
        if show:
            plt.show()


class Histogram:
    """Class for plots of ensembles of distributions as CDF (cumulative distribution function) or cCDF (complementary
        cumulative distribution function) with mean, median, and quantiles"""

    def __init__(self, sample_x):
        self.sample_x = sample_x

    def plot(
        self,
        ax=None,
        ylabel="PDF(x)",
        xtralabel="",
        xlabel="x",
        num_bins=50,
        show=False,
        outputname=None,
        color="C2",
        logscale=False,
        xlims=None,
    ):
        """Method to compile the plot. The plot is added to a provided matplotlib axes object or a new one is created.
            Arguments:
                ax: matplitlib axes             - the system of coordinates into which to plot
                ylabel: str                     - y axis label
                xlabel: str                     - x axis label
                num_bins: int                   - number of bins
                show: bool                      - show plot
                outputname: str                 - output file name without ending
                color: str or other admissible matplotlib color label - color to use for the plot
                logscale: bool                  - y axis logscale
                xlims: tuple, array of len 2, or none - x axis limits
            Returns: None."""

        """Create figure if none was provided"""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        """Plot"""
        ax.hist(self.sample_x, bins=num_bins, color=color)

        """Set plot attributes"""
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if xtralabel != "":
            anchored_text = AnchoredText(xtralabel, loc=1)
            ax.add_artist(anchored_text)

        """Set xlim if requested"""
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])

        """Set yscale to log if requested"""
        if logscale:
            ax.set_yscale("log")

        """Save if filename provided"""
        if outputname is not None:
            plt.savefig(outputname + ".pdf")
            plt.savefig(outputname + ".png", density=300)

        """Show if requested"""
        if show:
            plt.show()


class RiskModelSpecificCompare:
    def __init__(
        self,
        infiletype="_premium.dat",
        refiletype="_reinpremium.dat",
        number_riskmodels=4,
    ):
        """Initialises class that plots the insurance and reinsurance data for all risk models for specified data.
            Accepts:
                infiletype: Type String. The insurance data to be plotted.
                refiletype: Type String. The reinsurance data to be plotted.
                number_riskmodels. Type Integer. The number of riskmodels used in the data.
        This initialises the class by taking the specififed data from its file, calculating its mean, median, and
        25/75th percentiles. then adding it to a dictionary. Defaults to premium data however can plot any data."""
        self.timeseries_dict = {}
        self.timeseries_dict["mean"] = {}
        self.timeseries_dict["median"] = {}
        self.timeseries_dict["quantile25"] = {}
        self.timeseries_dict["quantile75"] = {}

        # Original filetypes is needed for the specific case of non insured risks in order to get correct y axis label.
        self.original_filetypes = self.filetypes = [infiletype, refiletype]
        self.number_riskmodels = number_riskmodels
        self.riskmodels = ["one", "two", "three", "four"]

        # Non insured risks is special as it needs contract data.
        uninsured_risks = False
        if self.filetypes[0] == "_noninsured_risks.dat":
            uninsured_risks = True
            num_risks = 20000
            self.filetypes = ["_contracts.dat", "_reincontracts.dat"]

        for i in range(number_riskmodels):
            for j in range(len(self.filetypes)):
                # Get required data out of file.
                filename = "data/" + self.riskmodels[i] + self.filetypes[j]
                rfile = open(filename, "r")
                data = [eval(k) for k in rfile]
                rfile.close()

                if (
                    uninsured_risks and j == 1
                ):  # Need operational data for calculating uninsured reinsurance risks.
                    rfile = open("data/" + self.riskmodels[i] + "_operational.dat", "r")
                    n_insurers = [eval(d) for d in rfile]
                    rfile.close()
                    n_pr = 4  # Number of peril categories

                # Compute data series
                data_means = []
                data_medians = []
                data_q25 = []
                data_q75 = []
                for k in range(len(data[0])):
                    if not uninsured_risks:  # Used for most risks.
                        data_means.append(np.mean([item[k] for item in data]))
                        data_q25.append(np.percentile([item[k] for item in data], 25))
                        data_q75.append(np.percentile([item[k] for item in data], 75))
                        data_medians.append(np.median([item[k] for item in data]))
                    elif (
                        uninsured_risks and j == 0
                    ):  # Used for uninsured insurance risks.
                        data_means.append(
                            np.mean([num_risks - item[k] for item in data])
                        )
                        data_q25.append(
                            np.percentile([num_risks - item[k] for item in data], 25)
                        )
                        data_q75.append(
                            np.percentile([num_risks - item[k] for item in data], 75)
                        )
                        data_medians.append(
                            np.median([num_risks - item[k] for item in data])
                        )
                    elif (
                        uninsured_risks and j == 1
                    ):  # Used for uninsured reinsurance risks.
                        data_means.append(
                            np.mean(
                                [
                                    n_pr * n_insurers[n][k] - data[n][k]
                                    for n in range(len(n_insurers))
                                ]
                            )
                        )
                        data_q25.append(
                            np.percentile(
                                [
                                    n_pr * n_insurers[n][k] - data[n][k]
                                    for n in range(len(n_insurers))
                                ],
                                25,
                            )
                        )
                        data_q75.append(
                            np.percentile(
                                [
                                    n_pr * n_insurers[n][k] - data[n][k]
                                    for n in range(len(n_insurers))
                                ],
                                75,
                            )
                        )
                        data_medians.append(
                            np.median(
                                [
                                    n_pr * n_insurers[n][k] - data[n][k]
                                    for n in range(len(n_insurers))
                                ]
                            )
                        )

                data_means = np.array(data_means)
                data_medians = np.array(data_medians)
                data_q25 = np.array(data_q25)
                data_q75 = np.array(data_q75)

                # Record data series
                self.timeseries_dict["mean"][filename] = data_means
                self.timeseries_dict["median"][filename] = data_medians
                self.timeseries_dict["quantile25"][filename] = data_q25
                self.timeseries_dict["quantile75"][filename] = data_q75

    def plot(self, outputfile):
        """Method to plot the insurance and reinsurance data for each risk model already initialised. Automatically
        saves data as pdf using argument provided.
            Accepts:
                outputfile: Type string. Used in naming of file to be saved to.
            No return values."""

        # List of colours and labels used in plotting
        colours = {"one": "red", "two": "blue", "three": "green", "four": "yellow"}
        labels = {
            "reinexcess_capital": "Excess Capital (Reinsurers)",
            "excess_capital": "Excess Capital (Insurers)",
            "cumulative_unrecovered_claims": "Uncovered Claims (cumulative)",
            "cumulative_bankruptcies": "Bankruptcies (cumulative)",
            "profitslosses": "Profits and Losses (Insurer)",
            "contracts": "Contracts (Insurers)",
            "cash": "Liquidity (Insurers)",
            "operational": "Active Insurers",
            "premium": "Premium (Insurance)",
            "reinprofitslosses": "Profits and Losses (Reinsurer)",
            "reincash": "Liquidity (Reinsurers)",
            "reincontracts": "Contracts (Reinsurers)",
            "reinoperational": "Active Reinsurers",
            "reinpremium": "Premium (Reinsurance)",
            "noninsured_risks": "Non-insured risks (Insurance)",
            "noninsured_reinrisks": "Non-insured risks (Reinsurance)",
        }

        # Backup existing figures (so as not to overwrite them)
        outputfilename = "figures/" + outputfile + "_riskmodel_comparison.pdf"
        backupfilename = (
            "figures/"
            + outputfile
            + "_riskmodel_comparison_old_"
            + time.strftime("%Y_%b_%d_%H_%M")
            + ".pdf"
        )
        if os.path.exists(outputfilename):
            os.rename(outputfilename, backupfilename)

        # Create figure and two subplot axes then plot on them using loop.
        self.fig, self.axs = plt.subplots(2, 1)
        for f in range(
            len(self.filetypes)
        ):  # Loop for plotting insurance then reinsurance data (length 2).
            maxlen_plots = 0
            for i in range(
                self.number_riskmodels
            ):  # Loop through number of risk models.
                # Needed for the fill_between method for plotting percentile data.
                if i <= 1:
                    j = 0
                else:
                    j = i - 1
                filename = "data/" + self.riskmodels[i] + self.filetypes[f]
                self.axs[f].plot(
                    range(len(self.timeseries_dict["mean"][filename]))[1200:],
                    self.timeseries_dict["mean"][filename][1200:],
                    color=colours[self.riskmodels[i]],
                    label=self.riskmodels[i] + " riskmodel(s)",
                )
                self.axs[f].fill_between(
                    range(
                        len(
                            self.timeseries_dict["quantile25"][
                                "data/" + self.riskmodels[j] + self.filetypes[f]
                            ]
                        )
                    )[1200:],
                    self.timeseries_dict["quantile25"][filename][1200:],
                    self.timeseries_dict["quantile75"][filename][1200:],
                    facecolor=colours[self.riskmodels[i]],
                    alpha=0.25,
                )
                maxlen_plots = max(
                    maxlen_plots, len(self.timeseries_dict["mean"][filename])
                )

            # Labels axes.
            xticks = np.arange(1200, maxlen_plots, step=600)
            self.axs[f].set_xticks(xticks)
            self.axs[f].set_xticklabels(
                ["${0:d}$".format(int((xtc - 1200) / 12)) for xtc in xticks]
            )
            ylabel = self.original_filetypes[f][1:-4]
            self.axs[f].set_ylabel(labels[ylabel])

        # Adds legend to top subplot and x axis label to bottom subplot.
        self.axs[0].legend(loc="best")
        self.axs[1].set_xlabel("Years")
        plt.tight_layout()

        # Saves figure and notifies user (no plt.show so allows progress tracking)
        self.fig.savefig(outputfilename)
        print("Have saved " + outputfile + " data")


if __name__ == "__main__":
    # Use argparse to handle command line arguments
    parser = argparse.ArgumentParser(description="Model the Insurance sector")
    parser.add_argument(
        "--single", action="store_true", help="plot a single run of the insurance model"
    )
    parser.add_argument(
        "--pie",
        action="store_true",
        help="plot animated pie charts of contract and cash data",
    )
    parser.add_argument(
        "--timeseries", action="store_true", help="plot time series of firm data"
    )
    parser.add_argument(
        "--timeseries_comparison",
        action="store_true",
        help="plot insurance and reinsurance "
        "time series for an ensemble of replications of "
        "the insurance model",
    )
    parser.add_argument(
        "--firmdistribution",
        action="store_true",
        help="plot the CDFs of firm size distributions with quartiles indicating variation across "
        "ensemble",
    )
    parser.add_argument(
        "--bankruptcydistribution",
        action="store_true",
        help="plot the histograms of bankruptcy events/unrecovered claims across ensemble",
    )
    parser.add_argument(
        "--riskmodel_comparison",
        action="store_true",
        help="Plot data comparing risk models for both insurance and reinsurance firms.",
    )
    args = parser.parse_args()

    if args.single:

        # load in data from the history_logs dictionary with open("data/history_logs.dat","r") as rfile:
        with open("data/history_logs.dat", "r") as rfile:
            history_logs_list = [eval(k) for k in rfile]  # one dict on each line

        # first create visualisation object, then create graph/animation objects as necessary
        vis = visualisation(history_logs_list)
        if args.pie:
            vis.insurer_pie_animation()
            vis.reinsurer_pie_animation()
        if args.timeseries:
            insurerfig, axs = vis.insurer_time_series()
            reinsurerfig, axs = vis.reinsurer_time_series()
            insurerfig.savefig("figures/insurer_singlerun_timeseries.png")
            reinsurerfig.savefig("figures/reinsurer_singlerun_timeseries.png")
        # vis.show()
        N = len(history_logs_list)

    if args.timeseries_comparison or args.bankruptcydistribution:
        vis_list = []
        colour_list = ["red", "blue", "green", "yellow"]

        # Loads all risk model history logs data (very long :'( ) and creates list of visualisation class instances.
        filenames = [
            "./data/" + x + "_history_logs.dat" for x in ["one", "two", "three", "four"]
        ]
        for filename in filenames:
            with open(filename, "r") as rfile:
                history_logs_list = [eval(k) for k in rfile]  # one dict on each line
                vis_list.append(visualisation(history_logs_list))

    if args.timeseries_comparison:
        # Creates time series for all risk models in ensemble data.
        cmp_rsk = compare_riskmodels(vis_list, colour_list)
        cmp_rsk.create_insurer_timeseries(percentiles=[10, 90])
        cmp_rsk.create_reinsurer_timeseries(percentiles=[10, 90])
        # cmp_rsk.show()

    if args.bankruptcydistribution:
        # Creates histogram for each number of risk models for size and frequency of bankruptcies/unrecovered claims.
        for vis in vis_list:
            vis.populate_scatter_data()

        HP = Histogram_plot(
            vis_list, colour_list, variable="bankruptcy_events_relative_clustered"
        )
        HP.generate_plot(
            logscale=True,
            xlabel="Share of bankrupt firms",
            minmax=[0, 0.5],
            VaR005guess=0.1,
        )  # =0.056338028169014086)    # this is the VaR threshold for 4 risk models with reinsuranc

        HP = Histogram_plot(vis_list, colour_list, variable="unrecovered_claims")
        HP.generate_plot(
            logscale=True,
            xlabel="Damages not recovered",
            minmax=[0, 6450000],
            VaR005guess=0.1,
        )  # =691186.8726311699)    # this is the VaR threshold for 4 risk models with reinsurance

    if args.firmdistribution:
        vis_list = []
        colour_list = ["red", "blue", "green", "yellow"]

        # Loads all risk model history logs data (very long :'( ) and creates list of visualisation class instances.
        filenames = [
            "./data/" + x + "_history_logs_complete.dat"
            for x in ["one", "two", "three", "four"]
        ]
        for filename in filenames:
            with open(filename, "r") as rfile:
                history_logs_list = [eval(k) for k in rfile]  # one dict on each line
                vis_list.append(visualisation(history_logs_list))

        # Creates CDF for firm size using cash as measure of size.
        CP = CDF_distribution_plot(
            vis_list,
            colour_list,
            variable="insurance_firms_cash",
            timestep=-1,
            plot_cCDF=True,
        )
        CP.generate_plot(xlabel="Firm size (capital)")
        if not isleconfig.simulation_parameters["reinsurance_off"]:
            CP = CDF_distribution_plot(
                vis_list,
                colour_list,
                variable="reinsurance_firms_cash",
                timestep=-1,
                plot_cCDF=True,
            )
            CP.generate_plot(xlabel="Firm size (capital)")

    if args.riskmodel_comparison:
        # Lists of insurance and reinsurance data files to be compared (Must be same size and equivalent).
        data_types = [
            "_noninsured_risks.dat",
            "_excess_capital.dat",
            "_cash.dat",
            "_contracts.dat",
            "_premium.dat",
            "_operational.dat",
        ]
        rein_data_types = [
            "_noninsured_reinrisks.dat",
            "_reinexcess_capital.dat",
            "_reincash.dat",
            "_reincontracts.dat",
            "_reinpremium.dat",
            "_reinoperational.dat",
        ]

        # Loops through data types and loads, plots, and saves each one.
        for type in range(len(data_types)):
            compare = RiskModelSpecificCompare(
                infiletype=data_types[type], refiletype=rein_data_types[type]
            )
            compare.plot(outputfile=data_types[type][1:-4])

# ਲ਼
