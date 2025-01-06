# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

"""ehist is a module for ergoonomic plotting of histograms. It aims to use sane defaults
based on the sample provided.

"""

__all__ = [
    "Hist1D",
    "Hist2D",
    "HistCat",
]

__version__ = "0.0.1"

from .hist1d import Hist1D
from .hist2d import Hist2D
from .hist_cat import HistCat
