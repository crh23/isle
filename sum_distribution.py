import numpy as np
import scipy.stats
import scipy.integrate
import scipy.interpolate
import distributiontruncated
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import multiprocessing as mp
import itertools

non_truncated = scipy.stats.pareto(b=2, loc=0, scale=0.25)

damage_distribution = distributiontruncated.TruncatedDistWrapper(
    lower_bound=0.25, upper_bound=1.0, dist=non_truncated
)


@functools.lru_cache(maxsize=4096)
def sum_beta_pdf(damage, n, resolution=5000):
    # Calculate the pdf of the sum of n beta variables
    x = np.linspace(start=0, stop=1, num=resolution)
    # the pdf of beta is inf at 1, so we slightly perturb the maximum point to avoid making a mess
    w = 5
    x[-1] = (w * x[-1] + x[-2]) / (w + 1)
    y = scipy.stats.beta(a=1, b=1 / damage - 1).pdf(x)
    # Turn y into a pmf
    y = y / sum(y)
    # Setting n=n * resolution - (n - 1) is very important
    pdf = np.fft.irfft(np.fft.rfft(y, n=n * resolution - (n - 1)) ** n)
    # Go from a pmf back to a pdf
    pdf = pdf * resolution
    x_new = np.linspace(start=0, stop=n, num=len(pdf))
    return scipy.interpolate.interp1d(
        x_new, pdf, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True
    )


def plot_mc(ns, no_mc_sims=100000, norm_approx=False, kde=True, hist=True):
    assert kde or hist
    print("Doing MC simulation for n = " + ", ".join([str(n) for n in ns]))
    for n in ns:
        results = []
        impacts = damage_distribution.rvs(no_mc_sims)
        for impact in impacts:
            if len(results) % 100 == 0:
                print(f"\rn = {n}: {len(results)/no_mc_sims:.1%}", end="")
            if not norm_approx:
                damage = sum(scipy.stats.beta(a=1, b=1 / impact - 1).rvs(size=n))
            else:
                damage = scipy.stats.norm(
                    loc=n * impact,
                    scale=np.sqrt(n * impact ** 2 * (1 - impact) / (1 + impact)),
                ).rvs()
            results.append(damage)
        print(f"\rn = {n}: 100.0%")
        results = np.array(results)
        x = np.linspace(start=results.min(), stop=results.max(), num=500)
        if kde:
            bw = 0.03
            approx_dist = scipy.stats.gaussian_kde(results, bw_method=bw)
            y = approx_dist.evaluate(x)
            plt.plot(x, y, label=f"Monte Carlo KDE, n = {n}")
        if hist:
            plt.hist(
                results,
                bins="auto",
                density=True,
                histtype="step",
                label=f"Monte Carlo Histogram, n = {n}",
            )


@functools.lru_cache(maxsize=4096)
def integrand(x, z, n):
    approx = n >= 20
    if approx:
        return scipy.stats.norm(
            loc=n * x, scale=np.sqrt(n * x ** 2 * (1 - x) / (1 + x))
        ).pdf(z) * damage_distribution.pdf(x)
    else:
        return sum_beta_pdf(x, n)(z) * damage_distribution.pdf(x)


def calc_pdf(z, n):
    return scipy.integrate.quad(integrand, 0.25, 1, args=(z, n))[0]


def make_save_pdf_data(ns):
    resolution = 500
    with mp.Pool(processes=round(mp.cpu_count() / 2)) as p:
        tasks = {}
        for n in ns:
            args = zip(
                np.linspace(start=0, stop=n, num=resolution), itertools.repeat(n)
            )
            tasks[n] = p.starmap_async(calc_pdf, args)
        results = {
            n: (np.linspace(start=0, stop=n, num=resolution), tasks[n].get())
            for n in ns
        }
    with open("./pdf_data.pkl", "wb") as wfile:
        pkl.dump(results, wfile, protocol=pkl.HIGHEST_PROTOCOL)


def plot_pdfs(pdfs, ns=None, normalise=False):
    if ns is None:
        ns = pdfs.keys()
    for n in ns:
        x = np.array(pdfs[n][0])
        y = np.array(pdfs[n][1])
        if normalise:
            x = x / n
            y = y * n
        plt.plot(x, y, label=f"Calculated pdf, n = {n}")


def plot_3d(pdfs):
    x = []
    y = []
    z = []
    for n in pdfs:
        assert len(pdfs[n][0]) == len(pdfs[n][1])
        x += list(np.array(pdfs[n][0]) / n)
        z += list(np.array(pdfs[n][1]) * n)
        y += [n] * len(pdfs[n][0])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print(len(x), len(y), len(z))
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x[::100], y[::100], z[::100], label="pdf surface")
    surf._facecolors2d = surf._facecolors3dp
    surf._edgecolors2d = surf._edgecolors3d


# ns = range(1, 2001)

# make_save_pdf_data(ns)

with open("./pdf_data.pkl", "rb") as rfile:
    pdfs: dict = pkl.load(rfile)
# ns = list(pdfs.keys())
ns = [10, 25, 50, 100, 200]
plot_pdfs(pdfs, ns, normalise=True)
x = np.linspace(start=0, stop=1, num=500)
plt.plot(x, damage_distribution.pdf(x), label="Base truncated Pareto")
# plot_mc(ns, kde=True, no_mc_sims=1000000)
plt.legend()
# plt.axis([0, 1050, 0, plt.axis()[3]])
plt.show()

# TODO: Probably shouldn't be using rightmost endpoint throughout (doesn't make much difference probably but is still wrong)
