import functools
import numpy as np
import scipy.stats
import warnings


class ReinsuranceDistWrapper:
    """ Wrapper for modifying the risk to an insurance company when they have EoL reinsurance

    dist is a distribution taking values in [0, 1] (as the damage distribution should) # QUERY: Check this
    lower_bound is the least reinsured risk (lowest priority), upper_bound is the greatest reinsured risk
    Note that the bounds are in terms of the values of the distribution, not the probabilities.

    Coverage is a list of tuples, each tuple representing a region that is reinsured. Coverage is in money, will be
    divided by value."""

    def __init__(
        self, dist, lower_bound=None, upper_bound=None, coverage=None, value=None
    ):
        if coverage is not None:
            if value is None:
                raise ValueError(
                    "coverage and value must both be passed or neither be passed"
                )
            if upper_bound is not None or lower_bound is not None:
                raise ValueError(
                    "lower_bound and upper_bound can't be used with coverage and value"
                )
        else:
            if value is not None:
                raise ValueError(
                    "coverage and value must both be passed or neither be passed"
                )
            if upper_bound is None and lower_bound is None:
                raise ValueError("no restriction arguments passed!")
        self.dist = dist
        if coverage is None:
            if lower_bound is None:
                lower_bound = 0
            elif upper_bound is None:
                upper_bound = 1
            assert 0 <= lower_bound < upper_bound <= 1
            self.coverage = [(lower_bound, upper_bound)]
        else:
            self.coverage = [
                (region[0] / value, region[1] / value) for region in coverage
            ]
        if self.coverage and self.coverage[0][0] == 0:
            warnings.warn("Adding reinsurance for 0 damage - probably not right!")
        # TODO: verify distribution bounds here
        # self.redistributed_share = dist.cdf(upper_bound) - dist.cdf(lower_bound)

    @functools.lru_cache(maxsize=512)
    def truncation(self, x):
        """ Takes a value x and returns the ammount of damage required for x damage to be absorbed by the firm.
        Also returns whether the value was on a boundary (point of discontinuity) (to make pdf, cdf work on edge cases)
        """
        # TODO: doesn't work with arrays, fix?
        if not np.isscalar(x):
            x = x[0]
        boundary = False
        for region in self.coverage:
            if x < region[0]:
                return x, boundary
            else:
                if x == region[0]:
                    boundary = True
                x += region[1] - region[0]
        return x, boundary

    def inverse_truncation(self, p):
        """ Returns the inverse of the above function, which is continuous and well-defined """
        # TODO: needs to work with arrays
        adjustment = 0
        for region in self.coverage:
            # These bounds are probabilities
            if p <= region[0]:
                return p - adjustment
            elif p < region[1]:
                return region[0] - adjustment
            else:
                adjustment += region[1] - region[0]
        return p - adjustment

    @functools.lru_cache(maxsize=512)
    def pdf(self, x):
        # derivative of truncation is 1 at all points of continuity, so only need to modify at boundaries
        result, boundary = self.truncation(x)
        if boundary:
            return np.inf
        else:
            return self.dist.pdf(result)

    @functools.lru_cache(maxsize=512)
    def cdf(self, x):
        # cdf is right-continuous modification, so doesn't care about the discontinuity
        result, _ = self.truncation(x)
        return self.dist.cdf(result)

    @functools.lru_cache(maxsize=512)
    def ppf(self, p):
        if type(p) is not float:
            p = p[0]
        return self.inverse_truncation(self.dist.ppf(p))

    def rvs(self, size=1):
        sample = self.dist.rvs(size=size)
        sample = map(self.inverse_truncation, sample)
        return sample


if __name__ == "__main__":
    # TODO: Check with coverage = []
    from distributiontruncated import TruncatedDistWrapper
    import matplotlib.pyplot as plt

    non_truncated = TruncatedDistWrapper(scipy.stats.pareto(b=2, loc=0, scale=0.5))
    # truncated = ReinsuranceDistWrapper(lower_bound=0, upper_bound=1, dist=non_truncated)
    truncated = ReinsuranceDistWrapper(
        dist=non_truncated, value=10, coverage=[(6.5, 7), (8, 9)]
    )

    x1 = np.linspace(non_truncated.ppf(0.01), non_truncated.ppf(0.99), 100).flatten()

    y1 = list(map(non_truncated.pdf, x1))
    y2 = list(map(truncated.pdf, x1))
    plt.plot(x1, y1, "r+")
    plt.plot(x1, y2, "bx")
    plt.legend(["non truncated", "truncated"])
    plt.show()

    # pdb.set_trace()
