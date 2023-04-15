# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

from ehist import Hist1D

values = range(1, 100)
h1 = Hist1D(values, t="logint")


print(dir(h1))
print(h1.A)
print(h1.N)
print(h1.x.widths)
print(h1.htext())
