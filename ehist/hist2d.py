# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pylab as plt
from matplotlib import colors

from .ansi_cmap import ansi_cmap


class Hist2D:
    def __init__(
        self,
        x,
        y,
        bins=(60, 50),
        range=None,  # noqa: A002
        log=(False, False),
        weights=1.0,
    ) -> None:
        assert len(x) == len(
            y,
        ), f"x and y must have the same dimensions: len(x) = {len(x)}, len(y) = {len(y)}"

        if np.isscalar(weights):
            weights = np.full_like(x, weights)

        assert len(x) == len(weights), (
            "weights, x, and y must have the same dimensions:"
            f"len(weights) = {len(weights)}, len(x) = {len(x)}, len(y) = {len(y)}"
        )

        self.logx, self.logy = log

        if range is None:
            rangex, rangey = None, None
        else:
            rangex, rangey = range

        if rangex is None:
            rangex = (np.min(x), np.max(x))
        if rangey is None:
            rangey = (np.min(y), np.max(y))

        if self.logx:
            x = np.log10(x)
            if rangex is not None:
                rangex = np.log10(rangex)

        if self.logy:
            y = np.log10(y)
            if rangey is not None:
                rangey = np.log10(rangey)

        if range is not None:
            range = (rangex, rangey)  # noqa: A001

        nancut = np.array(np.isfinite(x) & np.isfinite(y), dtype=bool)
        self.entries = nancut.sum()
        self.z, self.xedges, self.yedges = np.histogram2d(
            x[nancut],
            y[nancut],
            bins=bins,
            range=range,
            weights=weights[nancut],
        )

        if self.logx:
            self.xedges = 10**self.xedges
        if self.logy:
            self.yedges = 10**self.yedges

    def plot(self, logz=False, norm=None, zrange=None):
        z = np.ma.masked_where(self.z == 0, self.z)
        if norm is None:
            pass
        elif norm == "x":
            for x in range(z.shape[0]):
                z[x, :] = z[x, :] / z[x, :].sum()
        elif norm == "y":
            for y in range(z.shape[1]):
                z[:, y] = z[:, y] / z[:, y].sum()
        else:
            msg = f"Unknown value for norm: {norm}"
            raise ValueError(msg)

        if zrange is None:
            zrange = (z.min(), z.max())
        if logz:
            norm = colors.LogNorm(vmin=zrange[0], vmax=zrange[1])
        else:
            norm = colors.Normalize(vmin=zrange[0], vmax=zrange[1])

        plt.pcolormesh(self.xedges, self.yedges, z.T, norm=norm)
        plt.colorbar()
        plt.xlim(self.xedges[0], self.xedges[-1])
        plt.ylim(self.yedges[0], self.yedges[-1])

        if self.logx:
            plt.xscale("log")
        if self.logy:
            plt.yscale("log")

    def plot_text(self):
        return ansi_cmap(self.xedges, self.yedges, self.z.T)

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"bins=({len(self.xedges) - 1},{len(self.yedges) - 1})"
            f"range=[[{self.xedges[0]:0.2f},{self.xedges[-1]:0.2f}],"
            f"[{self.yedges[0]:0.2f},{self.yedges[-1]:0.2f}]]"
            f"entries={self.entries}>"
        )
