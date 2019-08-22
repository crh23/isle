import pickle as pkl
from typing import Tuple

import scipy.stats
import scipy.interpolate
import scipy.signal
import numpy as np
from genericclasses import RiskProperties

with open("./pdf_data.pkl", "rb") as rfile:
    pdfs: dict = pkl.load(rfile)


def find_nearest(array, value):
    """Given an array and a value, returns the element of the array nearest to the value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    """Given an array and a value, returns the index of the element of the array nearest to the value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class AggregateDistribtion(scipy.stats.rv_continuous):
    """
    Overall damage distribution for n risks in the same category. Distribution is normalised to lie between 0 and 1,
    so needs to be mulitplied by total_value
    """

    def __init__(self, n):
        super().__init__(a=0, b=1)
        self.n = n
        try:
            self._pdf_data = pdfs[n]
            self._pdf_x = np.array(self._pdf_data[0]) / self.n
            self._pdf_y = np.array(self._pdf_data[1]) * self.n
        except KeyError:
            n_fake = find_nearest(pdfs.keys(), self.n)
            print(
                f"Warning: tried to use non-existant pre-computed distribution for n = {self.n}"
            )
            print(f"Using scaled version of closest available value, {n_fake}")
            self._pdf_data = pdfs[n_fake]
            self._pdf_x = np.array(self._pdf_data[0]) / n_fake
            self._pdf_y = np.array(self._pdf_data[1]) * n_fake

        # The stored data usually isn't perfectly normalised due to numerical inaccuracy / floating point errors
        self._pdf_y = self._pdf_y / np.sum(self._pdf_y) * len(self._pdf_x)

        self._cdf_y = np.cumsum(self._pdf_y) / len(self._pdf_x)

        self._pdf = scipy.interpolate.interp1d(
            self._pdf_x,
            self._pdf_y,
            kind="cubic",
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

        self._cdf = scipy.interpolate.interp1d(
            self._pdf_x,
            self._cdf_y,
            kind="linear",
            bounds_error=False,
            fill_value=(0, 1),
            assume_sorted=True,
        )

    def _rvs(self, *args):
        raise Exception("Shouldn't generate rvs from this, there are better ways...")


def contract_risk(
    number_risks: int,
    value_per_risk: int,
    deductible: int,
    limit: int,
    runtime: int,
    max_claims: int,
    var_tail_size: float = 0.005,
    cat_frequency: float = 3 / 100,
    max_res: int = 2000,
    approx: bool = True,
) -> Tuple[float, float, int, int]:
    """
    Calculates various risk statistics for a contract.
    Uses the pre-calculated density functions for aggregate claims, so should be accurate even for only a few risks.
    If the total value (number_risks * value_per_risk) is very high (>10,000) then the calculations will be less
    accurate (not by much), as the calculations will be scaled for performance reasons. This is controled by max_res
    MU = monetary units
    Args:
        number_risks: The number of risk being (re)insured
        value_per_risk: The value of each risk (all risks have the same value) in MU
        deductible: The deductible of the contract under consideration in MU
        limit: The limit of the contract under consideration in MU
        runtime: The total runtime of the contract
        max_claims: The maximum number of claims that can be made. If 0, will be assumed equal to runtime.
        var_tail_size: The tail size to use when calculating the VaR (default 0.005)
        cat_frequency: The probability of a catastrophe in each time unit (default 3/100)
        max_res: The maximum number of value increments used in the calculations. Should be at least 1000

    Returns:
        expected_total_claim: The expectation of the total claims under this contract
        std: The standard deviation of the total claims
        var: The VaR in MU at the tail size given
        exposure: limit - deductible
    """

    # We assume throughout that claims are only whole-number monetary units, and are the ceiling of the actual damage.
    total_value = number_risks * value_per_risk

    # We change value_per_risk to get the resolution of the calculations to a suitable scale
    if total_value > max_res:
        factor = total_value / max_res
        total_value = int(round(total_value / factor))
        deductible = int(round(deductible / factor))
        limit = int(round(limit / factor))
    else:
        factor = 0

    claim_amounts = np.arange(total_value + 1)
    damage_probabilities = AggregateDistribtion(number_risks).pdf(
        claim_amounts / total_value
    )
    # Change from a density to a pmf
    damage_probabilities = damage_probabilities / np.sum(damage_probabilities)
    claim_probabilities = np.zeros_like(damage_probabilities)
    claim_probabilities[0 : limit - deductible] = damage_probabilities[deductible:limit]
    claim_probabilities[0] += np.sum(damage_probabilities[0:deductible])
    claim_probabilities[limit - deductible] += np.sum(damage_probabilities[limit:])
    # plt.plot(claim_amounts, damage_probabilities)
    # plt.plot(claim_amounts, claim_probabilities)
    # plt.show()

    if not (0 < max_claims <= runtime):
        max_claims = runtime

    per_event_exposure = min(limit, total_value) - deductible
    exposure = per_event_exposure * max_claims

    total_claim_values = np.arange(exposure + 1)
    total_claim_probabilities = np.zeros(total_claim_values.shape)
    for number_claims in range(max_claims + 1):
        p = scipy.stats.binom.pmf(n=runtime, k=number_claims, p=cat_frequency)
        if approx and p < 0.01:
            # Skip if probability of this number of events is close to trivial
            continue
        if number_claims == max_claims:
            # Add on all the events where there are more cats than the max number of claims
            p += scipy.stats.binom.sf(n=runtime, k=number_claims, p=cat_frequency)

        if number_claims == 0:
            # [1, 0, 0, ...]
            dist_this_number = np.array([1.0] + [0] * exposure)
        else:
            dist_this_number = claim_probabilities.copy()
            if number_claims > 1:
                dist_this_number = fftconvolve_n(claim_probabilities, number_claims)
        if len(dist_this_number) > len(total_claim_probabilities):
            total_claim_probabilities += (
                p * dist_this_number[: len(total_claim_probabilities)]
            )
        else:
            total_claim_probabilities[: len(dist_this_number)] += p * dist_this_number

    # Try to minimise the effect of numerical error and approximation by normalising
    total_claim_probabilities = total_claim_probabilities / np.sum(
        total_claim_probabilities
    )
    total_claim_cumulative_probabilities = np.cumsum(total_claim_probabilities)

    if total_claim_cumulative_probabilities[-1] >= 1 - var_tail_size:
        var = total_claim_values[
            np.searchsorted(
                total_claim_cumulative_probabilities, 1 - var_tail_size, side="left"
            )
        ]
    else:
        raise RuntimeError

    expected_total_claim = np.dot(total_claim_values, total_claim_probabilities)
    expected_square_total_claim = np.dot(
        total_claim_values ** 2, total_claim_probabilities
    )
    variance = expected_square_total_claim - expected_total_claim ** 2
    std = np.sqrt(variance)

    assert max(round(expected_total_claim), var) <= exposure
    if factor:
        expected_total_claim *= factor
        var = round(var * factor)
        exposure = round(exposure * factor)
        std = round(std * factor)

    return float(expected_total_claim), float(std), int(var), int(exposure)


def get_contract_risk(
    risk: RiskProperties, params: dict, max_claims=0
) -> Tuple[float, float, int, int]:
    tail_size = params["value_at_risk_tail_probability"]
    cat_sep = params["event_time_mean_separation"]
    for prop in [
        "number_risks",
        "value",
        "deductible_fraction",
        "limit_fraction",
        "runtime",
    ]:
        assert risk.__dict__[prop] is not None
    assert risk.deductible_fraction < risk.limit_fraction <= 1
    return contract_risk(
        number_risks=risk.number_risks,
        value_per_risk=int(round(risk.value / risk.number_risks)),
        deductible=int(round(risk.deductible_fraction * risk.value)),
        limit=int(round(risk.limit_fraction * risk.value)),
        runtime=risk.runtime,
        max_claims=max_claims,
        var_tail_size=tail_size,
        cat_frequency=1 / cat_sep,
    )


def fftconvolve_n(inp, n, axes=None):
    """
    Uses fft to convolve an array with itself n times.
    This function is a modification of scipy.signal.fftconvolve.
    Takes an input of a single numpy array-like and convolves it with itself n times using a fast fourier transform.

    Args:
        inp:
        n:
        axes:

    Returns:

    """
    from scipy.fftpack.helper import _init_nd_shape_and_axes_sorted
    import scipy.fftpack as fftpack

    inp = np.asarray(inp)

    if inp.ndim == 0:  # scalar input
        return inp ** n
    elif inp.size == 0:
        return np.array([])

    _, axes = _init_nd_shape_and_axes_sorted(inp, shape=None, axes=axes)

    if axes is not None and not axes.size:
        raise ValueError("when provided, axes cannot be empty")

    shape = np.array(inp.shape)
    shape[axes] = shape[axes] * n - (n - 1)  # This is vital!

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [fftpack.helper.next_fast_len(d) for d in shape[axes]]
    fslice = tuple([slice(sz) for sz in shape])

    sp = np.fft.rfftn(inp, fshape, axes=axes)
    ret = np.fft.irfftn(sp ** n, fshape, axes=axes)[fslice].copy()

    return ret


def main():
    import matplotlib.pyplot as plt

    for _ in range(100):
        risks = []
        for res in np.logspace(start=1, stop=5, num=50):
            risks.append(
                contract_risk(
                    number_risks=200,
                    value_per_risk=10000,
                    deductible=100 * 10000,
                    limit=200 * 10000,
                    runtime=12,
                    max_claims=1,
                    max_res=int(res),
                )
            )
    # risks = np.asarray(risks)
    # plt.plot(np.logspace(start=1, stop=5, num=50), risks, "rx")
    # for statistic in risks[-1]:
    #     plt.axhline(y=statistic, lw=1)
    # plt.xscale('log')
    # plt.legend()
    # plt.show()

    # import distributiontruncated
    #
    # est_dist = distributiontruncated.TruncatedDistWrapper(
    #     lower_bound=0.25,
    #     upper_bound=1.0,
    #     dist=scipy.stats.pareto(b=2, loc=0, scale=0.25),
    # )
    # # lb = 1
    # # ub = 100
    # # ns = range(lb, ub)
    # # vars = []
    # # for n in ns:
    # #     vars.append(AggregateDistribtion(n=n).ppf(1 - 0.025))
    # # plt.plot(range(lb, ub), vars, label="VaR for each n")
    # # plt.plot(range(lb, ub), [est_dist.ppf(1 - 0.025) for _ in range(lb, ub)],
    #               label="Estimated VaR using truncated Pareto")
    # ns = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    # x = np.linspace(0, 1, num=100)
    #
    # for n in ns:
    #     dist = AggregateDistribtion(n=n)
    #     plt.plot(x, dist.cdf(x), label=f"n = {n}")
    #
    # plt.plot(x, est_dist.cdf(x), label="Truncated Pareto")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
