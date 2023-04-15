# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pylab as plt
from scipy import stats

lims = (stats.norm.cdf(-1), stats.norm.cdf(1))

print(lims)
cl = lims[-1] - lims[0]


x = np.arange(11)

e1 = x**0.5
e2 = x**0.5

N = stats.norm.interval(0.68, loc=x, scale=e1)
print(N)

plt.errorbar(x, x, yerr=[e1, e2], ls="none", marker="o")

plt.plot(x, N[1], c="red")
plt.plot(x, N[0], c="red")


P = stats.poisson.interval(0.68, x)
print(P)
plt.plot(x, P[0], c="g")
plt.plot(x, P[1], c="g")

plt.show()
