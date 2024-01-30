
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
    # ts = (xbar - ybar) / np.sqrt((sx**2 / nx) + (sy**2 / ny))
    a = xbar - ybar
    b = np.sqrt((((nx - 1) * sx**2) + ((ny - 1) * sy**2)) / (nx + ny - 2))
    c = np.sqrt((1 / nx) + (1 / ny))
    ts = a / (b * c)
    t_pdf = lambda x: t.pdf(x, df=nx + ny - 2, loc=0, scale=1)

    return ts, quad(t_pdf, ts, np.inf)[0]

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

def example_means():
    """
    compare mean womens height (x) and mens height (y) based on data samples
    """

    # define x and y samples
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

def example_proportions():
    """
    compare proportions (conversion rates at price A and at price B) based on data samples
    -- outcome --   -- price A --   -- price B --
    conversion          200             182
    no conversion      23539           22406
    """

    # define conversion data at price A and at price B
    conv_a = np.hstack((np.ones(200), np.zeros(23539)))
    conv_b = np.hstack((np.ones(182), np.zeros(22406)))

    # get sample proportions at price A and at price B
    pa_sample_proportion = conv_a.sum() / conv_a.size
    pb_sample_proportion = conv_b.sum() / conv_b.size

    # get 95% confidence intervals for population proportions at price A and at price B
    pa_population_proportion = get_conf_interval_bootstrap(conv_a, confidence=0.95)
    pb_population_proportion = get_conf_interval_bootstrap(conv_a, confidence=0.95)

    # get p-value for conversion rate at price A > conversion rate at price B, i.e.
    # H0: pa_population_proportion <= pb_population_proportion
    # HA: pa_population_proportion > pb_population_proportion
    pvalue_t = get_pvalue_t_statistic(conv_a, conv_b)
    pvalue_b = get_pvalue_bootstrap(conv_a, conv_b)
