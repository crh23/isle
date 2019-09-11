import numpy as np
import scipy.stats as stats


def calculate_single(log: dict, t: int = -1):
    """Takes a dict as returned by Logger.obtainlog() and returns a vector of statistics
    We always look at year-long data"""

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

    insvars = [ins_pls, ins_assets, ins_claims, ins_premiums]
    revars = [re_pls, re_assets]

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

    output = []
    for data in (ins_pls, ins_assets, ins_claims, ins_premiums):
        st = stats.describe(data, nan_policy="raise")
        for result in (st.mean, st.variance, st.skewness, st.kurtosis):
            output.append(result)

    for data in (re_pls, re_assets):
        st = stats.describe(data, nan_policy="raise")
        for result in (st.mean, st.variance):
            output.append(result)

    """Next do market premium"""
    premium = np.mean(log["market_premium"][t - 12 : t])
    output.append(premium)
    return output
