# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pylab as plt
from scipy import integrate

from ehist import Hist1D


def powerlaw(x, a, b):
    return a * x**b


x = np.geomspace(0.1, 100, 1000)

hf = Hist1D(x, t="log")
hf.plot(logy=True)
print(hff := hf.fit(powerlaw, (1, 1)))
plt.plot(x, powerlaw(x, *hff[0]))

print(integrate.quad(powerlaw, x[0], x[-1], args=tuple(hff[0])))

print(x.astype(int))


plt.figure()

hi = Hist1D(x.astype(int), t="logint")
hi.plot("err", logy=True)
print(hif := hi.fit(powerlaw, (1, 1)))
plt.plot(x, powerlaw(x, *hif[0]))

print(hi.htext(logy=True))
print(hi.vtext(logy=True))

print(hi.N)

print(integrate.quad(powerlaw, x[0], x[-1], args=tuple(hif[0])))

plt.show()

"""
Npoints = int(1e4)
a = np.linspace(0,1,Npoints)

funcs = [
    (lambda x: 2*x    , lambda x, a, b: a*x+b,[1,5e3]),
    (lambda x: 2* x**.5, lambda x, a, b: a*x+b,[5e3,6]),
    #(stats.norm.ppf, lambda x, a, b, c: a* np.exp( -(x-b)**2/2/c) , [1e4,1,2]),
]


for trans, fit,p0 in funcs:

    plt.figure()
    h=Hist1D(trans(a))


    print(h.N.sum(),Npoints)
    print('A', h.A.sum())
    print('H',h.H)
    h.plot('err'    )
    cf = h.fit(fit,p0)
    print('I',integrate.quad( fit, h.x.edges[0],h.x.edges[-1], args=tuple(cf[0])))
    print("Hr", h.H/h.Herr)
    px = h.x.uniform()
    plt.plot(px, fit(px,*p0),ls=':')
    plt.plot(px, fit(px,*cf[0]))



    #print(integrate.quad(fit, h.x.edges[0], h.x.edges[1],args=tuple(p0)))

    ifit = h.fit(fit, p0, method='quad')
    print('PFIT', cf)
    print('IFIT', ifit)


    plt.plot(px, fit(px,*ifit.x))
    """
plt.show()
