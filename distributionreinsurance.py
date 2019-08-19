from genericclasses import weak_lru_cache
import numpy as np
import scipy.stats
import warnings


class ReinsuranceDistWrapper:
    """
    Wrapper for modifying the risk to an insurance company when they have EoL reinsurance.
    The reinsurance can be given either by passing upper_bound and lower_bound or by passing coverage.

    Args:
        dist: the distribution to modify. Must be a probability distribution on [0, 1]
        lower_bound: the lower bound of reinsurance, given as a proportion of the possible damage
        upper_bound: the upper bound of reinsurance, given as a proportion of the possible damage
        coverage: the reinsurance coverage, given as the actual values reinsured. Is a list of tuples, each tuple
            representing a region of coverage. Values will be divided by value
        value: the total value of the reinsured assets, used to divide coverage
    """

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
            self.coverage = [(lower_bound, upper_bound)]
        else:
            self.coverage = [
                (region[0] / value, min(region[1] / value, 1))
                for region in coverage
                if region[0] < value
            ]
        for region in self.coverage:
            assert 0 <= region[0] < region[1] <= 1

        if self.dist.cdf(0) != 0 or self.dist.cdf(1) != 1:
            raise ValueError(
                "Distribution passed is not bounded 0 <= x <= 1, so is not supported"
            )

    @weak_lru_cache(maxsize=512)
    def truncation(self, x):
        """ Takes an array-like x and returns the ammount of damage required for x damage to be absorbed by the firm.
        Also returns whether the value was on a boundary (point of discontinuity) (to make pdf, cdf both work on edge
        cases)
        """
        # Need to take a copy of x as we will be working directly on it
        x = np.array(x)

        boundary = np.zeros_like(x, dtype=bool)
        for region in self.coverage:
            if (x <= region[0]).all():
                break
            else:
                boundary[x == region[0]] = True
                x[x > region[0]] += region[1] - region[0]
        return x, boundary

    @weak_lru_cache(maxsize=512)
    def inverse_truncation(self, p):
        """ Returns the inverse of the above function, which is continuous and well-defined """

        # Don't need to copy p, since no modifications are made to it
        p = np.asarray(p)
        adjustment = np.zeros_like(p)
        for region in self.coverage:
            # These bounds are probabilities
            if (p < region[0]).all():
                break
            adjustment[region[0] <= p < region[1]] += (
                p[region[0] <= p < region[1]] - region[0]
            )
            adjustment[p >= region[1]] += region[1] - region[0]
        return p - adjustment

    @weak_lru_cache(maxsize=512)
    def pdf(self, x):
        # derivative of truncation is 1 at all points of continuity, so only need to modify at boundaries
        result, boundary = self.truncation(x)
        to_return = np.zeros_like(result)
        to_return[np.logical_not(boundary)] = self.dist.pdf(
            result[np.logical_not(boundary)]
        )
        to_return[boundary] = np.inf
        return to_return

    @weak_lru_cache(maxsize=512)
    def cdf(self, x):
        # cdf is right-continuous modification, so doesn't care about the discontinuity
        result, _ = self.truncation(x)
        return self.dist.cdf(result)

    @weak_lru_cache(maxsize=512)
    def ppf(self, p):
        p = np.asarray(p)
        return self.inverse_truncation(self.dist.ppf(p))

    def rvs(self, size=None):
        sample = self.dist.rvs(size=size)
        sample = self.inverse_truncation(sample)
        if size is None:
            return sample[0]
        else:
            return sample


if __name__ == "__main__":
    from distributiontruncated import TruncatedDistWrapper
    import matplotlib.pyplot as plt

    non_truncated = TruncatedDistWrapper(scipy.stats.pareto(b=2, loc=0, scale=0.5))
    # truncated = ReinsuranceDistWrapper(lower_bound=0, upper_bound=1, dist=non_truncated)
    truncated = ReinsuranceDistWrapper(
        dist=non_truncated, value=10, coverage=[(6, 6.25), (7, 7.5)]
    )

    x1 = np.linspace(non_truncated.ppf(0.01), non_truncated.ppf(0.99), 100)

    y1 = non_truncated.pdf(x1)
    y2 = truncated.pdf(x1)
    plt.plot(x1, y1, "r+", label="non truncated")
    plt.plot(x1, y2, "bx", label="truncated")
    plt.legend()
    plt.show()

    # pdb.set_trace()
