
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrb.mpl import open_figure, save_pngs, format_axes, largefonts
from scipy.optimize import curve_fit
plt.style.use('bmh')


fleet = [100, 300, 500, 700]
coverage = {}
coverage['10sec'] = [0.449, 1.667, 2.803, 3.702]
coverage['30sec'] = [1.48, 4.00, 6.22, 7.96]
coverage['60sec'] = [2.33, 6.44, 9.86, 12.32]
coverage['5min'] = [8.76, 19.56, 26.46, 29.92]

def func(x, a, b):
    return a * x + b

fig, ax = open_figure('extrapolate coverage', figsize=(12, 6))
for key in coverage.keys():
    p = ax.plot(fleet, coverage[key], 'x-', ms=12, lw=4, label=key)[0]
    (a, b), _ = curve_fit(func, np.array(fleet), np.array(coverage[key]))
    ax.plot(np.array(fleet), func(np.array(fleet), a, b), '--', lw=4, color=p.get_color())
format_axes('fleet size, thousand vehicles', 'collision coverage by nearby vehicles, %', 'extrapolate collision coverage by nearby vehicles', ax)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, handlelength=5)
largefonts(18)
fig.tight_layout()
plt.show()
