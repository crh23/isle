from scipy import stats
import numpy as np


class ConstantGen(stats.rv_continuous):
    def _pdf(self, x, *args):
        a = np.float_(x == 0)
        a[a == 1.0] = np.float_("inf")
        return a

    def _cdf(self, x, *args):
        return np.float_(x >= 0)

    def _rvs(self, *args):
        if self._size is None:
            return 0.0
        else:
            return np.zeros(shape=self._size)


constant = ConstantGen(name="constant")
