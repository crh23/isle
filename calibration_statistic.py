import numpy as np
import scipy.stats as stats
import scipy.signal as sig

# A list of the statistics that are outputted (for convenience). This is the canonical order.
# statistics = [
#     "ins_profitloss_mean",
#     "ins_profitloss_var",
#     "ins_profitloss_skew",
#     "ins_profitloss_kurt",
#     "ins_assets_mean",
#     "ins_assets_var",
#     "ins_assets_skew",
#     "ins_assets_kurt",
#     "ins_claims_mean",
#     "ins_claims_var",
#     "ins_claims_skew",
#     "ins_claims_kurt",
#     "ins_premiums_mean",
#     "ins_premiums_var",
#     "ins_premiums_skew",
#     "ins_premiums_kurt",
#     "ins_solvency_II_mean",
#     "ins_solvency_II_var",
#     "ins_solvency_II_skew",
#     "ins_solvency_II_kurt",
#     "re_profitloss_mean",
#     "re_profitloss_var",
#     "re_assets_mean",
#     "re_assets_var",
#     "re_solvency_II_mean",
#     "re_solvency_II_var",
#     "rate_on_line",
# ]

# A list of the statistics we actually have data for
statistics = [
    "ins_profitloss_mean",
    "ins_profitloss_var",
    "ins_profitloss_skew",
    "ins_profitloss_kurt",
    "ins_assets_mean",
    "ins_assets_var",
    "ins_assets_skew",
    "ins_assets_kurt",
    "ins_claims_mean",
    "ins_claims_var",
    "ins_claims_skew",
    "ins_claims_kurt",
    "ins_premiums_mean",
    "ins_premiums_var",
    "ins_premiums_skew",
    "ins_premiums_kurt",
    "re_profitloss_mean",
    "re_profitloss_var",
    "re_assets_mean",
    "re_assets_var",
    "rate_on_line_autocorr_lag_1",
    "rate_on_line_autocorr_lag_2",
    "rate_on_line_autocorr_lag_3",
    "rate_on_line_autocorr_lag_4",
    "rate_on_line_autocorr_lag_5",
    "rate_on_line_autocorr_lag_6",
    "rate_on_line_autocorr_lag_7",
    "rate_on_line_autocorr_lag_8",
    "rate_on_line_autocorr_lag_9",
    "rate_on_line_autocorr_lag_10",
]

# The observed data
observed = {
    "ins_profitloss_mean": 760.8403583654745,
    "ins_profitloss_var": 2645861.077652064,
    "ins_profitloss_skew": 4.2449962851289325,
    "ins_profitloss_kurt": 19.079106920165742,
    "ins_assets_mean": 34422.302595397,
    "ins_assets_var": 3760288749.1412654,
    "ins_assets_skew": 2.493880852639668,
    "ins_assets_kurt": 5.3551029266263885,
    "ins_claims_mean": 4256.6503666102135,
    "ins_claims_var": 116386072.62414846,
    "ins_claims_skew": 4.3147875953543515,
    "ins_claims_kurt": 17.92245449535894,
    "ins_premiums_mean": 3938.0877987962162,
    "ins_premiums_var": 24680436.944519438,
    "ins_premiums_skew": 1.5042246765191107,
    "ins_premiums_kurt": 0.9007423845542157,
    "re_profitloss_mean": 604.9962383992475,
    "re_profitloss_var": 362488.40656840097,
    "re_assets_mean": 30621.369781367932,
    "re_assets_var": 2159558210.9914727,
    "rate_on_line_autocorr_lag_1": 29.812898800240387,
    "rate_on_line_autocorr_lag_2": 28.600956884326997,
    "rate_on_line_autocorr_lag_3": 27.489320890771207,
    "rate_on_line_autocorr_lag_4": 25.83402370785473,
    "rate_on_line_autocorr_lag_5": 24.36635259681405,
    "rate_on_line_autocorr_lag_6": 23.029625203121448,
    "rate_on_line_autocorr_lag_7": 21.708364947181977,
    "rate_on_line_autocorr_lag_8": 20.715011404172028,
    "rate_on_line_autocorr_lag_9": 19.940742580799636,
    "rate_on_line_autocorr_lag_10": 19.133566632717763,
}


def calculate_single(log: dict, t: int = -1) -> dict:
    """Takes a dict as returned by Logger.obtainlog() and returns a vector of statistics
    We always look at year-long data"""

    print("Calculating calibration statistic...")
    """First do firm-wise timeseries data"""
    ins_pls = np.array(
        [sum(firm_data[t - 12 : t]) for firm_data in log["insurance_pls"]]
    )
    re_pls = np.array(
        [sum(firm_data[t - 12 : t]) for firm_data in log["reinsurance_pls"]]
    )
    ins_assets = np.array([firm_data[t] for firm_data in log["insurance_firms_cash"]])
    re_assets = np.array([firm_data[t] for firm_data in log["reinsurance_firms_cash"]])
    ins_claims = np.array(
        [sum(firm_data[t - 12 : t]) for firm_data in log["insurance_claims"]]
    )
    ins_premiums = np.array(
        [firm_data[t] for firm_data in log["insurance_cumulative_premiums"]]
    )
    ins_ratios = np.array([firm_data[t] for firm_data in log["insurance_ratios"]])
    re_ratios = np.array([firm_data[t] for firm_data in log["reinsurance_ratios"]])

    insvars = [ins_pls, ins_assets, ins_claims, ins_premiums, ins_ratios]
    revars = [re_pls, re_assets, re_ratios]

    for vars_ in (insvars, revars):
        if not all([len(x) == len(vars_[0]) for x in vars_]):
            raise ValueError("Data are not all same length")

    ins_mask = ins_assets > 0
    re_mask = re_assets > 0

    ins_pls = ins_pls[ins_mask]
    re_pls = re_assets[re_mask]
    ins_assets = ins_assets[ins_mask]
    re_assets = re_assets[re_mask]
    ins_claims = ins_claims[ins_mask]
    ins_premiums = ins_premiums[ins_mask]
    ins_ratios = ins_ratios[ins_mask]
    re_ratios = re_ratios[re_mask]

    ins_data = {
        "ins_profitloss": ins_pls,
        "ins_assets": ins_assets,
        "ins_claims": ins_claims,
        "ins_premiums": ins_premiums,
        "ins_solvency_II": ins_ratios,
    }
    re_data = {
        "re_profitloss": re_pls,
        "re_assets": re_assets,
        "re_solvency_II": re_ratios,
    }
    statistic_names = ["mean", "var", "skew", "kurt"]

    output = {}
    for name, data in ins_data.items():
        if len(data) == 0:
            if len(ins_mask) == 0:
                print(f"Empty ins data found, name = {name}")
                for stat_name in statistic_names:
                    output[name + "_" + stat_name] = float("nan")
            else:
                for stat_name in statistic_names:
                    output[name + "_" + stat_name] = 0
        else:
            st = stats.describe(data, nan_policy="raise", ddof=0)
            for stat, stat_name in zip(
                (st.mean, st.variance, st.skewness, st.kurtosis), statistic_names
            ):
                output[name + "_" + stat_name] = stat

    for name, data in re_data.items():
        if len(data) == 0:
            if len(re_mask == 0):
                print(f"Empty re data found, name = {name}")
                for stat_name in statistic_names:
                    output[name + "_" + stat_name] = float("nan")
            else:
                for stat_name in statistic_names:
                    output[name + "_" + stat_name] = 0
        else:
            st = stats.describe(data, nan_policy="raise", ddof=0)
            for stat, stat_name in zip((st.mean, st.variance), statistic_names):
                output[name + "_" + stat_name] = stat

    """Next do market premium (Rate-On-Line)"""
    no_years = len(log["market_premium"]) // 4 // 12
    slices = [slice(-(n + 1) * 12, -n * 12) for n in range(no_years - 1, -1, -1)]
    slices[-1] = slice(-12, None, None)
    premium_series = np.array([np.mean(log["market_premium"][sl]) for sl in slices])
    premium_series = premium_series / np.mean(premium_series)
    ac = sig.correlate(premium_series, premium_series, mode="full")
    ac = ac[len(ac) // 2 :]
    for lag, c in enumerate(ac[:11]):
        if lag >= 1:
            output["rate_on_line_autocorr_lag_" + str(lag)] = c
    return output


def make_from_excel():
    import pandas as pd

    data = pd.read_excel("Data.xlsx", sheet_name=None)
    transposed_data = {}
    for key in data:
        df = data[key].T
        transposed_data[key.replace("(", " ").replace(")", " ")] = df.rename(
            columns=df.iloc[0]
        ).drop(df.index[0])
    ins = transposed_data["Insurance remove inflation "].loc[2014]
    re = transposed_data["Reinsurance remove inflation "].loc[2014]
    oth = transposed_data["Other Data"].loc[2014]

    ins_pls = np.array(ins.iloc[1:37], dtype=np.float_)
    re_pls = np.array(re.iloc[1:13], dtype=np.float_)
    ins_assets = np.array(ins.iloc[39:75], dtype=np.float_)
    re_assets = np.array(re.iloc[15:27], dtype=np.float_)
    ins_claims = np.array(ins.iloc[104:130], dtype=np.float_)
    ins_premiums = np.array(ins.iloc[77:103], dtype=np.float_)
    # ins_ratios =
    # re_ratios =

    ins_data = {
        "ins_profitloss": ins_pls,
        "ins_assets": ins_assets,
        "ins_claims": ins_claims,
        "ins_premiums": ins_premiums,
    }
    re_data = {"re_profitloss": re_pls, "re_assets": re_assets}

    statistic_names = ["mean", "var", "skew", "kurt"]

    output = {}
    for name, data in ins_data.items():
        st = stats.describe(data, nan_policy="raise", ddof=0)
        for stat, stat_name in zip(
            (st.mean, st.variance, st.skewness, st.kurtosis), statistic_names
        ):
            output[name + "_" + stat_name] = stat

    for name, data in re_data.items():
        st = stats.describe(data, nan_policy="raise", ddof=0)
        for stat, stat_name in zip((st.mean, st.variance), statistic_names):
            output[name + "_" + stat_name] = stat

    rols = list(
        transposed_data["Other Data"]["Guy Carpenter U.S. Property Rate on Line Index"]
    )[28::-1]
    rols = rols / np.mean(rols)
    ac = sig.correlate(rols, rols, mode="full")
    ac = ac[len(ac) // 2 :]
    for lag, c in enumerate(ac[:11]):
        if lag >= 1:
            output["rate_on_line_autocorr_lag_" + str(lag)] = c
    print(repr(output))


if __name__ == "__main__":
    make_from_excel()
