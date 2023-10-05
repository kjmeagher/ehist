#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

import sys

import numpy as np
import pylab as plt
from scipy import stats  # pylint: disable=import-error

from ehist import Hist1D as h1


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
        h.x.uniform()
    plt.show()


def triangle():
    x = stats.triang.ppf(np.linspace(0, 1, 1000, 1), 1)
    return {"points": x, "bins": 120}, {"s": "steps,err"}


def power_law():
    e = np.geomspace(1e3, 1e6, 1000)
    return {"points": e, "bins": 30, "t": "log"}, {"s": "err,marker,steps", "logy": True}


def zenith():
    cz = np.arccos(np.linspace(-1, 1, 1000))
    return {"points": cz, "bins": 20, "t": np.cos}, {"s": "err,marker,steps"}


def integer():
    x = np.linspace(0, 21, 106)[:-1].astype(int)
    return {"points": x, "bins": 20, "t": int}, {"s": "err,marker,steps"}


def log_int():
    n = 1001
    x = [int(n / i) * [i] for i in range(1, n)]
    p = [val for sublist in x for val in sublist]
    return {"points": p, "t": "logint", "bins": 20}, {"s": "steps,marker,err", "logy": True}


def freedman():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return {"points": x, "bins": "freedman"}, {}


def scott():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return {"points": x, "bins": "scott"}, {}


def knuth():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return {"points": x, "bins": "knuth"}, {}


def blocks():
    x = stats.norm.ppf(np.linspace(0, 1, 1002)[1:-1])
    return {"points": x, "bins": "blocks"}, {}


if __name__ == "__main__":
    main(locals())
