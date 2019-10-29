
"""
multiple regression toy problem and visualizations
"""

from pyrb import datasets

# load regression datasets
rbo = datasets.regression_boston()
rca = datasets.regression_california()
rdi = datasets.regression_diabetes()
rco = datasets.regression_concrete()
rde = datasets.regression_demand()
rtr = datasets.regression_traffic()

# test that residuals of regression follow a normal distribution
