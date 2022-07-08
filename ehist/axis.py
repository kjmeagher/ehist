"""
axis.py

This file contains different implementations of histgram axes
"""

import math as m
import numbers

import numpy as np
from matplotlib.ticker import FixedLocator

from .bayesian_blocks import bayesian_blocks, freedman_bin_width, knuth_bin_width, scott_bin_width
from .util import geomspace_int


class IntAxis:
    """
    Axis for formatting integer histograms
    """

    def bin_spacing(self, points, nbins, span):
        if span is None:
            span = [int(np.min(points)), int(np.max(points))]
        else:
            span = list(span)

        assert len(span) == 2
        assert isinstance(span[0], numbers.Integral)
        assert isinstance(span[1], numbers.Integral)
        span[1] += 1
        binsize = int(np.ceil((span[1] - span[0]) / nbins))

        # first try to align the bins
        low_bin = binsize * int(span[0] / binsize)
        high_bin = binsize * int(np.ceil(span[1] / binsize))
        self.bins = np.arange(low_bin, high_bin + 1, binsize)

        # if that crates too many bins start at the first bin instead
        if len(self.bins) > nbins + 1:
            low_bin = int(span[0])
            high_bin = low_bin + nbins * binsize
            self.bins = np.arange(low_bin, high_bin + 1, binsize)

        assert len(self.bins) <= nbins + 1
        assert all(issubclass(type(b), np.integer) for b in self.bins)
        self.bins[0] = span[0]
        self.bins[-1] = span[1]

    def finish(self):
        assert np.issubdtype(self.bins.dtype.type, np.integer)
        self.widths = self.bins[1:] - self.bins[:-1]
        if np.all(self.widths == 1):
            self.pcenters = self.bins[:-1]
            self.pedges = self.bins - 0.5
            self.edges = self.bins - 0.5
        else:
            self.pcenters = (self.bins[1:] + self.bins[:-1]) / 2
            self.pedges = self.bins
            self.edges = self.bins

    def pylab_axis(self, axis):

        if np.all(self.widths == 1):
            axis.xaxis.set_tick_params(which="major", length=0)
            x = self.pcenters
        else:
            x = self.bins

        axis.xaxis.set_minor_locator(FixedLocator(self.pedges))
        if self.pedges.size > 10:
            xi = list(range(0, int(0.9 * len(x)), max(1, len(x) // 5))) + [len(x) - 1]
        else:
            xi = range(len(x))
        axis.xaxis.set_major_locator(FixedLocator(x[xi]))

    def uniform(self):
        return np.arange(self.bins[0], self.bins[-1])


class LogIntAxis:
    """
    Axis for ploting integer distribution which are logarythmic

    This creates an axis with bins spaced approximately geometrically but
    rounded to the nearest integer. Special consideration is take for low
    numbers where geometric spaceing will lead to more than one bin per
    integer.
    """

    def bin_spacing(self, points, bins, range_):

        if range_ is None:
            range_ = [int(np.min(points)), int(np.max(points))]

        if np.isscalar(bins):
            self.bins = geomspace_int(range_[0], range_[1], bins)
        else:
            self.bins = bins

    def finish(self):
        assert np.issubdtype(self.bins.dtype.type, np.integer)
        self.widths = self.bins[1:] - self.bins[:-1]
        self.pcenters = np.sqrt(self.bins[1:] * self.bins[:-1])
        self.pedges = self.bins
        self.edges = self.bins - 0.5

    def pylab_axis(self, ax):
        ax.set_xscale("log")

    def uniform(self):
        return np.arange(self.bins[0], self.bins[-1])


class LinearAxis:
    def bin_spacing(self, points, bins, range_):
        if range_ is None:
            range_ = [np.min(points), np.max(points)]
        self.bins = np.linspace(range_[0], range_[1], bins + 1)

    def finish(self):
        self.edges = self.bins
        self.widths = self.edges[1:] - self.edges[:-1]
        self.pcenters = (self.edges[1:] + self.edges[:-1]) / 2
        self.pedges = self.edges

    def pylab_axis(self, ax):
        pass

    def uniform(self):
        if len(self.bins) >= 100:
            return self.bins
        return np.linspace(self.bins[0], self.bins[-1], 100)


class LogAxis:
    def bin_spacing(self, points, bins, range_):
        if range_ is None:
            range_ = [np.min(points[points > 0]), np.max(points)]
        self.bins = np.geomspace(range_[0], range_[1], bins + 1)

    def finish(self):
        self.edges = self.bins
        self.widths = self.edges[1:] - self.edges[:-1]
        self.pcenters = np.sqrt(self.edges[1:] * self.edges[:-1])
        self.pedges = self.edges

    def pylab_axis(self, ax):
        ax.set_xscale("log")

    def uniform(self):
        if len(self.bins) >= 100:
            return self.bins
        return np.geomspace(self.bins[0], self.bins[-1], 100)


class ZenithAxis:
    def bin_spacing(self, points, bins, range_):
        if range_ is None:
            range_ = [np.min(points), np.max(points)]
        # coerse the values to common edges
        if range_[0] < 0.1:
            range_[0] = 0
        if range_[1] >= np.pi / 2 - 0.1 and range_[1] <= np.pi / 2:
            range_[1] = np.pi / 2
        if range_[1] >= np.pi - 0.1 and range_[1] <= np.pi:
            range_[1] = np.pi

        self.range = range_
        self.cosbins = np.linspace(np.cos(self.range[0]), np.cos(self.range[1]), bins + 1)
        self.bins = np.arccos(self.cosbins)

    def finish(self):
        self.edges = self.bins
        self.widths = self.cosbins[:-1] - self.cosbins[1:]
        self.pcenters = (self.cosbins[1:] + self.cosbins[:-1]) / 2
        self.pedges = self.cosbins

    def pylab_axis(self, ax):
        span = self.range[1] - self.range[0]
        if span > np.deg2rad(span):
            major = np.array([0, 45, 60, 75, 90, 105, 120, 135, 180])
        else:
            major = np.array([0, 30, 45, 60, 75, 90, 105, 120, 135, 150, 180])
        rmajor = np.deg2rad(major)
        cut = (rmajor >= self.range[0] * 0.99) & (rmajor <= self.range[1] * 1.0001)
        minor = range(0, 181, 5)
        ax.xaxis.set_ticks(np.cos(rmajor[cut]))
        ax.xaxis.set_ticklabels(str(m) + "°" for m in major[cut])
        ax.xaxis.set_minor_locator(FixedLocator(np.cos(np.deg2rad(minor))))

    def uniform(self):
        if len(self.bins) >= 100:
            return self.bins
        return np.cos(np.linspace(self.bins[0], self.bins[-1], 100))


def AutoAxis(points, bins, range_, t):

    points = np.asarray(points)

    if t is None:
        if np.issubdtype(points.dtype.type, np.integer):
            t = int
        elif issubclass(points.dtype.type, np.floating):
            t = float
        else:
            raise ValueError(f"don't know what to do with points dtype={points.dtype}")

    if t in [int, "int"]:
        ax = IntAxis()
    elif t in ["logint"]:
        ax = LogIntAxis()
    elif t in [np.log, np.log10, m.log, m.log10, "log"]:
        ax = LogAxis()
    elif t in [np.cos, m.cos, "cos"]:
        ax = ZenithAxis()
    elif t in [float, None]:
        ax = LinearAxis()
    else:
        raise ValueError

    if isinstance(bins, str):
        if bins == "blocks":
            bins = bayesian_blocks(points)
        elif bins == "knuth":
            _, bins = knuth_bin_width(points, True)
        elif bins == "scott":
            _, bins = scott_bin_width(points, True)
        elif bins == "freedman":
            _, bins = freedman_bin_width(points, True)
        else:
            raise ValueError(f"unrecognized bin code: '{bins}'")
    elif bins is None:
        bins = 64

    if np.isscalar(bins):
        ax.bin_spacing(points, bins, range_)
    else:
        ax.bins = bins

    ax.finish()
    return ax


def print_axis(axis):
    print(
        f"{axis.__class__.__name__}\n\tb: {axis.bins}\n\tw: {axis.widths}\n\te: {axis.edges}"
        f"\n\tc: {axis.pcenters}\n\te: {axis.pedges}"
    )
