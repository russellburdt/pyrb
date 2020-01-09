
"""
functions for inferential statistics

- x is a data sample
- y is a data sample (may or may not be from same distribution as x)

- 95% confidence interval for population mean ux or uy can be determined by
  * get_conf_interval_t_statistic
  * get_conf_interval_bootstrap

- 95% confidence interval for the difference of population means (uy - ux) can be determined by
  * get_conf_interval_t_statistic_xy
  * get_conf_interval_bootstrap_xy

- in two cases above, t-statistic method and bootstrap methods give same answer
  * bootstrap method is easier to understand and carries less assumptions

- p-value for hypothesis that x is greater than y can be determined by
  * get_pvalue_t_statistic_xy
  * get_pvalue_bootstrap_xy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.integrate import quad
from pyrb import open_figure, format_axes, largefonts
from ipdb import set_trace
from tqdm import tqdm
plt.style.use('bmh')

def get_conf_interval_t_statistic(x, confidence=0.95):
    """
    return confidence interval for a random variable x based on t-distribution
    """

    # sample statistics
    xbar = np.mean(x)
    sx = np.std(x)
    nx = len(x)

    # confidence interval based on t-distribution
    alpha = (1 - confidence) / 2
    ts = np.abs(t.ppf(q=alpha, df=nx - 1, loc=0, scale=1))
    b = ts * sx / nx**0.5

    return xbar - b, xbar + b

def get_conf_interval_bootstrap(x, confidence=0.95, n=10000):
    """
    return confidence interval for a random variable x based on bootstrap resampling
    """

    # get sample size and initialize a list for bootstrap sample means
    nx = x.size
    xb = []
    for _ in tqdm(range(n), desc='bootstrap confidence interval'):
        xb.append(np.mean(np.random.choice(x, replace=True, size=nx)))

    # return a confidence interval based on the bootstrapped samples
    alpha = 100 * (1 - confidence) / 2
    return np.percentile(xb, alpha), np.percentile(xb, 100 - alpha)

def get_conf_interval_t_statistic_xy(x, y, confidence=0.95):
    """
    return confidence interval for a random variable (x - y) based on t-distribution
    """

    # sample statistics
    xbar, ybar = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    nx, ny = len(x), len(y)

    # confidence interval based on t-distribution
    alpha = (1 - confidence) / 2
    ts = np.abs(t.ppf(q=alpha, df=nx + ny - 2, loc=0, scale=1))
    b = ts * np.sqrt((sx**2 / nx) + (sy**2 / ny))

    return (xbar - ybar) - b, (xbar - ybar) + b

def get_conf_interval_bootstrap_xy(x, y, confidence=0.95, n=10000):
    """
    return confidence interval for a random variable (x - y) based on bootstrap resampling
    """

    # get sample sizes and combine data from both samples
    nx, ny = x.size, y.size
    data = np.hstack((x, y))

    # initalize a list for bootstrapped differences of sample means
    bootstrap = []
    for _ in tqdm(range(n), desc='bootstrap confidence interval'):
        xbar = np.mean(np.random.choice(x, replace=True, size=nx))
        ybar = np.mean(np.random.choice(y, replace=True, size=ny))
        bootstrap.append(xbar - ybar)

    # return a confidence interval based on the bootstrapped samples
    alpha = 100 * (1 - confidence) / 2
    return np.percentile(bootstrap, alpha), np.percentile(bootstrap, 100 - alpha)

def get_pvalue_t_statistic(x, y):
    """
    return p-value for hypothesis test that mean of population x exceeds mean of population y
    - x is a data sample from population x
    - y is a data sample from population y
    - H0: ux <= uy
    - HA: ux > uy
    """

    # get sample statistics
    nx, ny = x.size, y.size
    xbar, ybar = x.mean(), y.mean()
    sx, sy = x.std(), y.std()

    # get t-statistic under null hypothesis (ux = uy)
    ts = (xbar - ybar) / np.sqrt((sx**2 / nx) + (sy**2 / ny))
    t_pdf = lambda x: t.pdf(x, df=nx + ny - 2, loc=0, scale=1)

    return quad(t_pdf, ts, np.inf)[0]

def get_pvalue_bootstrap(x, y, n=10000):
    """
    return p-value for hypothesis test that mean of population x exceeds mean of population y
    - x is a data sample from population x
    - y is a data sample from population y
    - H0: ux <= uy
    - HA: ux > uy
    """

    # get initial sample difference
    diff = x.mean() - y.mean()

    # get sample sizes and combine data from both samples
    nx, ny = x.size, y.size
    data = np.hstack((x, y))
    idx = range(data.size)

    # initalize a list for bootstrap differences of sample means
    bootstrap = []
    for _ in tqdm(range(n), desc='bootstrap confidence interval'):
        x_idx = np.random.choice(idx, replace=False, size=nx)
        y_idx = np.array(list(set(idx).difference(x_idx)))
        bootstrap.append(data[x_idx].mean() - data[y_idx].mean())
    bootstrap = np.array(bootstrap)

    # p-value is the number of times the bootstrap difference exceeds the original difference
    return (bootstrap > diff).sum() / n

def example():

    # example - compare womens (x) and mens (y) heights based on data samples
    x = np.array([5.33, 5.33, 5.17, 5.75, 5.42, 5.42, 5.50, 5.50, 5.58, 5.33, 5.50, 5.67, 5.42, 5.25,
                  6.17, 5.42, 5.33, 5.17, 5.42, 5.42, 5.42, 5.42, 5.42, 5.83, 5.33, 5.67, 5.33, 5.66,
                  5.25, 5.75, 5.57, 5.35, 5.42, 5.08, 5.75, 5.33, 5.08])
    y = np.array([5.75, 5.92, 6.17, 6.08, 5.58, 5.92, 6.00, 5.75, 5.92, 5.75, 5.75, 5.83, 6.58, 6.00,
                  5.75, 6.42, 6.50, 6.17, 6.00, 5.67, 5.58, 5.83, 5.58, 5.58, 6.08, 5.67, 6.00])
    y -= 0.45

    # get 95% confidence intervals for population mean heights
    ux_95_t = get_conf_interval_t_statistic(x, confidence=0.95)
    ux_95_b = get_conf_interval_bootstrap(x, confidence=0.95)
    uy_95_t = get_conf_interval_t_statistic(y, confidence=0.95)
    uy_95_b = get_conf_interval_bootstrap(y, confidence=0.95)

    # get 95% confidence interval for population mean mens height minus womens height
    uy_minus_ux_95_t = get_conf_interval_t_statistic_xy(y, x, confidence=0.95)
    uy_minus_ux_95_b = get_conf_interval_bootstrap_xy(y, x, confidence=0.95)

    # get p-value for mens height > womens height, i.e. H0: uy <= ux, HA: uy > ux
    pvalue_t = get_pvalue_t_statistic(y, x)
    pvalue_b = get_pvalue_bootstrap(y, x)

# # find symmetric bounds of normal distribution with 'confidence' of finding a random variable
# confidence = 0.98
# norm_pdf = lambda x: norm.pdf(x, loc=0, scale=1)
# norm_integral = lambda x: quad(norm_pdf, -x, x)[0]
# bound = fsolve(lambda x: norm_integral(x) - confidence, 0)

# # find probability of getting X heads in n flips of a fair coin
# X = 4900
# n = 10000
# print('prob of {} heads in {} flips of fair coin - {:.3f}'.format(X, n, binom(n=n, p=0.5).pmf(X)))
# print('prob of {} or more heads in {} flips of fair coin - {:.3f}'.format(X, n, binom(n=n, p=0.5).pmf(range(X, n + 1)).sum()))
# xbar = X / n
# z = (xbar - 0.50) / ((0.50 * (1 - 0.50)) / n)**0.5
# pvalue = (1 - norm_integral(z)) / 2
# print('prob of {} or more heads in {} flips of fair coin - {:.3f}'.format(X, n, pvalue))

# determine if data are from a normal distribution
# N = 1000
# normal = np.random.normal(4, 2, size=N)
# not_normal = np.hstack((np.random.normal(2, 4, size=int(N/2)), np.random.normal(2, 1, size=int(N/2))))
# p_normal = normaltest(normal).pvalue
# p_not_normal = normaltest(not_normal).pvalue
# jbp_normal = jarque_bera(normal)[1]
# jbp_not_normal = jarque_bera(not_normal)[1]
# title_normal = 'normal data' + \
#                '\nnull hypothesis: x is from normal distribution' + \
#                '\nscipy normaltest p-value={:g}'.format(p_normal) + \
#                '\nscipy Jarque-Bera p-value={:g}'.format(jbp_normal)
# title_not_normal = 'not normal data' + \
#                '\nnull hypothesis: x is from normal distribution' + \
#                '\nnormaltest p-value={:g}'.format(p_not_normal) + \
#                '\nJarque-Bera p-value={:g}'.format(jbp_not_normal)
# fig, ax = open_figure('scipy.stats.normaltest', 1, 2, figsize=(12, 6), sharex=True)
# ax[0].hist(normal, bins=np.linspace(-10, 10, 100))
# ax[1].hist(not_normal, bins=np.linspace(-10, 10, 100))
# format_axes('x', 'bin count', title_normal, ax[0])
# format_axes('x', 'bin count', title_not_normal, ax[1])
# fig.tight_layout()
# plt.show()


