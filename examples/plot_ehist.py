#!/usr/bin/env python
import sys

import numpy as np
import pylab as plt
from scipy import stats

from ehist import Hist1D as h1
from ehist.axis import print_axis


def main(d):
    opts = [x.lower() for x in sys.argv[1:]]
    for x, y in d.items():
        if len(sys.argv) > 1 and x.lower() not in opts:
            continue
        if x in ["main", "h1", "print_axis"]:
            continue
        if not callable(y):
            continue
        hargs, pargs = y()
        h = h1(**hargs)
        plt.figure()
        h.plot(**pargs)
        plt.title(x)
        xx = h.x.uniform()
    plt.show()


def Triangle():
    x = stats.triang.ppf(np.linspace(0, 1, 1000, 1), 1)
    return dict(points=x, bins=120), dict(s="steps,err")


def PowerLaw():
    e = np.geomspace(1e3, 1e6, 1000)
    return dict(points=e, bins=30, t="log"), dict(s="err,marker,steps", logy=True)


def Zenith():
    cz = np.arccos(np.linspace(-1, 1, 1000))
    return dict(points=cz, bins=20, t=np.cos), dict(s="err,marker,steps")


def Int():
    X = np.linspace(0, 21, 106)[:-1].astype(int)
    return dict(points=X, bins=20, t=int), dict(s="err,marker,steps")


def LogInt():
    N = 1001
    x = [int(N / i) * [i] for i in range(1, N)]
    p = [val for sublist in x for val in sublist]
    return dict(points=p, t="logint", bins=20), dict(s="steps,marker,err", logy=True)


def Freedman():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return dict(points=x, bins="freedman"), {}


def Scott():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return dict(points=x, bins="scott"), {}


def Knuth():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return dict(points=x, bins="knuth"), {}


def Blocks():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return dict(points=x, bins="blocks"), {}


if __name__ == "__main__":
    main(locals())
