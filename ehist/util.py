# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

import numbers

import numpy as np


def span(x):
    return x[-1] - x[0]


def is_integer(x):
    """Return true if parameter is integer type or float that exactly represents and integer."""
    return isinstance(x, numbers.Integral) or x.is_integer()


def geomspace_int(start, stop, num=50, dtype=np.int64):
    """Return integers approximately spaced evenly on a log scale
    (a geometric progression).

    This is similar to `numpy.geomspace()` except it returns integers.

    Parameters
    ----------
    start : integer
        The starting value of the sequence.
    stop : integer
        The final value of the sequence.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will always be
        an integer.
    """
    if num == 0:
        return np.array([], dtype=dtype)
    if num < 0:
        msg = f"Number of samples, {num}, must be non-negative"
        raise ValueError(msg)
    if not is_integer(start) or not is_integer(stop) or not is_integer(num):
        msg = "Parameters start, stop, and num are required to be integers"
        raise ValueError(msg)
    if not np.issubdtype(dtype, np.integer):
        msg = "Parameters dtype must be an integer type"
        raise ValueError(msg)

    start = int(start)
    stop = int(stop)

    if stop < start:
        tmp = stop
        stop, start, order = start, tmp, -1
    else:
        order = 1

    if start == 0:
        start = 1
        result = [0]
    else:
        result = []
    result += [start]
    last_delta = 1

    while len(result) < num and result[-1] < stop:
        ratio = (stop / result[-1]) ** (1.0 / (num - len(result)))
        delta = max(last_delta, int(np.round(result[-1] * (ratio - 1))))
        result.append(result[-1] + delta)
    result[-1] = stop

    return np.array(result, dtype=dtype)[::order]


def handle_weights(w, weights):
    # Handle the weights
    if w is not None and weights is not None:
        msg = "you can not set `w` and `weights` at the same time"
        raise ValueError(msg)
    if weights is None:
        weights = w
    if weights is None:
        weighted = False
        weights = 1
        scaled = False
    elif np.isscalar(weights):
        weighted = False
        scaled = True
    else:
        weighted = True
        scaled = False
        weights = np.array(weights, dtype=float)
    return weights, weighted, scaled


def get_name(obj):
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)


block_chars = " ▏▎▍▌▋▊▉█"
line_char = "━"
arrow_char = "→"
pm_char = "±"

vertical_blocks = " ▁▂▃▄▅▆▇█"
vertical_line = "┃"


def make_bar(y, min_y, max_y, width, blocks):
    if max_y <= min_y:
        raise ValueError
    if len(blocks) < 2:  # noqa: PLR2004
        raise ValueError
    empty_block = blocks[0]
    frac_blocks = blocks[1:-1]
    full_block = blocks[-1]

    if y <= min_y:
        bar = width * empty_block
    elif y >= max_y:
        bar = width * full_block
    else:
        mul = len(frac_blocks) + 1
        xx = round(float(y - min_y) * width * mul / (max_y - min_y))
        mod = int(xx % mul)
        bar = int(xx / mul) * full_block
        if mod and frac_blocks:
            bar += frac_blocks[mod - 1]
        bar += (width - len(bar)) * empty_block
    assert len(bar) == width
    return bar


class HorizontalPlot:
    def __init__(self, width=80) -> None:
        self.cols = []
        self.rows = None
        self.width = width

    def add_column(self, col, align="<", t=str, prefix="", postfix=""):
        if self.rows is None:
            self.rows = len(col)
        else:
            assert self.rows == len(col)

        if t in [float]:
            a = np.log10(max(np.abs(col))) > 5  # noqa: PLR2004
            b = min(np.abs(col)) != 0
            c = np.log10(min(np.abs(col[col != 0]))) < -2  # noqa: PLR2004
            fmt = "{:7.2e}" if a or b and c else "{:0.2f}"
            c = [fmt.format(r) for r in col]
            align = ">"
        elif t in [int]:
            align = ">"
            c = [str(r) for r in col]
        else:
            c = [str(r) for r in col]

        m = max(len(r) for r in c)
        self.cols.append((c, m, align, prefix, postfix))

    def get_plot(self, y, min_y, max_y):
        assert self.rows == len(y)
        bar_width = self.width - sum(m + len(pre) + len(post) + 1 for _, m, _, pre, post in self.cols)
        line = self.width * line_char

        out = line + "\n"

        for i in range(self.rows):
            out += " ".join(f"{pre}{c[i]:{a}{m}}{post}" for c, m, a, pre, post in self.cols)
            out += " " + make_bar(y[i], min_y, max_y, bar_width, block_chars)
            out += "\n"
        out += line + "\n"
        return out


class VerticalPlot:
    def __init__(self, width=80, height=25) -> None:
        self.width = width
        self.height = height

    def get_plot(self, y, min_y, max_y):
        print(min_y, max_y)
        rep = max(1, self.width // len(y))
        bars = []
        for yy in y:
            bars += rep * [make_bar(yy, min_y, max_y, self.height, vertical_blocks)[::-1]]
        return "".join("".join(a) + "\n" for a in zip(*bars))
