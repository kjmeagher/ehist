# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

_lower_block = "▄"
_reset_color = "\u001b[0m"
_jet = [
    21,
    27,
    33,
    39,
    45,
    51,
    50,
    49,
    48,
    47,
    46,
    82,
    118,
    154,
    190,
    226,
    220,
    214,
    208,
    202,
    196,
]
_viridis = [53, 54, 55, 61, 67, 74, 80, 79, 78, 77, 113, 149, 185, 226]


def setfg(color):
    return "\u001b[38;5;" + str(color) + "m"


def setbg(color):
    return "\u001b[48;5;" + str(color) + "m"


def ansi_cmap(x, y, z, cmap=None):
    assert x.size in [z.shape[1], x.size == z.shape[1] + 1]
    assert y.size in [z.shape[0], y.size == z.shape[0] + 1]

    cmap = _viridis.copy() if cmap is None else cmap.copy()

    zmin = z[z > 0].min()
    ci = (z - zmin) / (z.max() - zmin) * (len(cmap))
    ci = ci.astype(int)
    ci = np.clip(ci, 0, len(cmap) - 1)
    ci[z == 0] = len(cmap)
    cmap += [15]

    x0l = f"{x[0]:0.3g}"
    x1l = f"{x[-1]:0.3g}"
    y0l = f"{y[0]:0.3g}"
    y1l = f"{y[-1]:0.3g}"
    margin = max(len(y0l), len(y1l))
    s = ""
    for yi in range(0, z.shape[0], 2)[::-1]:
        if yi == 0:
            s += f"{y0l:>{margin}}"
        elif yi >= z.shape[0] - 2:
            s += f"{y1l:>{margin}}"
        else:
            s += margin * " "
        if yi + 1 >= z.shape[0]:
            for xi in range(z.shape[1]):
                s += setfg(cmap[ci[yi, xi]]) + _lower_block
        else:
            for xi in range(z.shape[1]):
                s += setfg(cmap[ci[yi, xi]]) + setbg(cmap[ci[yi + 1, xi]]) + _lower_block
        s += _reset_color + "\n"

    s += margin * " " + x0l + (z.shape[1] - len(x0l) - len(x1l)) * " " + x1l
    return s
