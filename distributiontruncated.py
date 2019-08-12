import scipy.stats
import numpy as np
from math import ceil
import scipy.integrate
import functools


class TruncatedDistWrapper:
    def __init__(self, dist, lower_bound=0.0, upper_bound=1.0):
        self.dist = dist
        self.normalizing_factor = dist.cdf(upper_bound) - dist.cdf(lower_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        assert self.upper_bound > self.lower_bound

    # @functools.lru_cache(maxsize=1024)
    def pdf(self, x):
        # TODO: begone, arrays
        x = np.array(x, ndmin=1)
        r = map(
            lambda y: self.dist.pdf(y) / self.normalizing_factor
            if (self.lower_bound <= y <= self.upper_bound)
            else 0,
            x,
        )
        r = np.array(list(r))
        if len(r.flatten()) == 1:
            r = float(r)
        return r

    # @functools.lru_cache(maxsize=1024)
    def cdf(self, x):
        # TODO: rm arrays
        x = np.array(x, ndmin=1)
        r = map(
            lambda y: 0
            if y < self.lower_bound
            else 1
            if y > self.upper_bound
            else (self.dist.cdf(y) - self.dist.cdf(self.lower_bound))
            / self.normalizing_factor,
            x,
        )
        r = np.array(list(r))
        if len(r.flatten()) == 1:
            r = float(r)
        return r

    # @functools.lru_cache(maxsize=1024)
    def ppf(self, x):
        # TODO: probably no need for arrays
        x = np.array(x, ndmin=1)
        assert (x >= 0).all() and (x <= 1).all()
        return self.dist.ppf(
            x * self.normalizing_factor + self.dist.cdf(self.lower_bound)
        )

    def rvs(self, size=1):
        # We could also use inverse transform sampling
        # Sample RVs from the original distribution and then throw out the ones that are outside the bounds.
        init_sample_size = int(ceil(size / self.normalizing_factor * 1.1))
        sample = self.dist.rvs(size=init_sample_size)
        sample = sample[
            np.logical_and(self.lower_bound <= sample, sample <= self.upper_bound)
        ]
        while len(sample) < size:
            sample = np.append(sample, self.rvs(size - len(sample)))
        return sample[:size]

    # Cache could be replaced with a simple "if is None" cache, might offer a small performance gain.
    # Also this could be a read-only @property, but then again so could a lot of things.
    @functools.lru_cache(maxsize=1)
    def mean(self):
        mean_estimate, mean_error = scipy.integrate.quad(
            lambda x: x * self.pdf(x), self.lower_bound, self.upper_bound
        )
        return mean_estimate


if __name__ == "__main__":
    non_truncated = scipy.stats.pareto(b=2, loc=0, scale=0.5)
    truncated = TruncatedDistWrapper(
        lower_bound=0.55, upper_bound=1.0, dist=non_truncated
    )

    x1 = np.linspace(non_truncated.ppf(0.01), non_truncated.ppf(0.99), 100)
    x2 = np.linspace(truncated.ppf(0.01), truncated.ppf(0.99), 100)

    print(truncated.mean())
