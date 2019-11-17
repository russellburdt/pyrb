
"""
implementation of basic statistical tests
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fsolve
from pyrb import open_figure, format_axes, largefonts


# find symmetric bounds of normal distribution with 'confidence' of finding a random variable
confidence = 0.98
norm_pdf = lambda x: norm.pdf(x, loc=0, scale=1)
norm_integral = lambda x: quad(norm_pdf, -x, x)[0]
bound = fsolve(lambda x: norm_integral(x) - confidence, 0)

# mean, sigma
baseline = (1367, 171)
n = 9
wc = (1419, 248)
dc = (1606, 158)








# import numpy as np
# import matplotlib.pyplot as plt
# from pyrb import open_figure, format_axes, largefonts
# from scipy.stats import norm, bernoulli, binom
# plt.style.use('bmh')

# # number of coin flips
# n = 100
# expt = np.array([np.random.randint(0, 2, n).sum() for _ in range(100000)])


# x = np.linspace(-10, 10, num=1000)

# fig, ax = open_figure('scipy.stats.bernoulli', 2, 1, sharex=True)
# ax[0].plot(x, binom.pmf(k=x, p=0.5, n=5), '-', lw=2)
# ax[1].plot(x, binom.cdf(k=x, p=0.5, n=5), '-', lw=2)
# format_axes('', '', 'scipy.stats.bernoulli.pmf', ax[0])
# format_axes('', '', 'scipy.stats.bernoulli.cdf', ax[1])
# largefonts(14)
# fig.tight_layout()

# fig, ax = open_figure('scipy.stats.norm', 2, 1, sharex=True)
# ax[0].plot(x, norm.pdf(x=x, loc=0, scale=1), '-', lw=2)
# ax[1].plot(x, norm.cdf(x=x, loc=0, scale=1), '-', lw=2)
# format_axes('', '', 'scipy.stats.norm.pdf', ax[0])
# format_axes('', '', 'scipy.stats.norm.cdf', ax[1])
# largefonts(14)
# fig.tight_layout()

# plt.show()



