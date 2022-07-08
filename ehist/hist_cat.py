import numpy as np
import pylab as plt

from .util import HorizontalPlot, arrow_char, get_name, handle_weights, pm_char


class HistCat:
    def __init__(self, data, w=None, weights=None, names=None, label=None, color=None):

        weights, self.weighted, self.scaled = handle_weights(w, weights)
        self.vals, self.N = np.unique(data, return_counts=True)
        if self.weighted:
            assert len(data) == len(weights)
            self.A = np.array([sum(w[data == v]) for v in self.vals])
            self.Aerr = np.array([sum(w[data == v] ** 2) ** 0.5 for v in self.vals])
        else:
            self.A = weights * self.N
            self.Aerr = weights * self.N**0.5

        self.size = len(self.vals)
        self.x = np.arange(self.size)

        if names is None:
            self.names = [str(xx) for xx in self.vals]
        elif callable(names):
            self.names = [get_name(names(xx)) for xx in self.vals]
        elif hasattr(names, "get"):
            self.names = [get_name(names.get(xx, xx)) for xx in self.vals]

        self.label = label
        self.color = color

    def plot(self, s="bars", rotation=0, logy=False, **kwargs):
        args = {"color": self.color, "label": self.label}
        args.update(kwargs)
        if s == "bars":
            plt.bar(self.x, self.A, tick_label=self.names, **args)

        elif s == "steps":
            x = np.arange(len(self.A) + 1) - 0.5
            plt.step(np.r_[x, x[-1]], np.r_[0, self.A, 0], where="pre", **args)
            ax = plt.gca()
            ax.set_xticks(self.x)
            ax.set_xticklabels(self.names)

        plt.xticks(rotation=rotation)
        if logy:
            plt.yscale("log")

    def plot_text(self, width=80, show_vals=True, show_counts=True, show_err=False, logy=False):
        bary = np.log(self.A) if logy else self.A
        min_y = min(bary) if logy else 0
        max_y = max(bary)
        p = HorizontalPlot(width=width)

        if show_vals:
            p.add_column(self.vals, ">")
        if self.names:
            p.add_column(self.size * [arrow_char])
            p.add_column(self.names)
        if show_counts:
            p.add_column(self.size * ":")
            p.add_column(self.A, t=float if self.weighted or self.scaled else int)
        if show_err:
            p.add_column(self.size * [pm_char])
            p.add_column(self.Aerr, t=float)
        return p.get_plot(bary, min_y, max_y)
