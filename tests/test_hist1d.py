# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

import functools
import operator
import unittest

import numpy as np
from numpy.testing import assert_allclose

from ehist import Hist1D


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


class TestAxis(unittest.TestCase):
    def test_integer(self):
        v = functools.reduce(operator.iadd, [i * [i] for i in range(10)], [])
        h = Hist1D(v)
        assert_allclose(h.N, range(1, 10))
        assert_allclose(h.H, range(1, 10))
        assert_allclose(h.Herr, np.arange(1, 10) ** 0.5)
        assert_allclose(h.A, range(1, 10))
        assert_allclose(h.Aerr, np.arange(1, 10) ** 0.5)
        assert_allclose(h.fit(linear, [0.5, 1])[0], [1, 0], atol=1e-6)

        h = Hist1D(v, range=(0, 9))
        assert_allclose(h.N, range(10))
        assert_allclose(h.H, range(10))
        assert_allclose(h.Herr, np.arange(10) ** 0.5)
        assert_allclose(h.A, range(10))
        assert_allclose(h.Aerr, np.arange(10) ** 0.5)
        assert_allclose(h.fit(linear, [0.5, 1])[0], [1, 0], atol=1e-6)

        h = Hist1D(v, weights=5)
        assert_allclose(h.N, range(1, 10))
        assert_allclose(h.H, 5 * np.arange(1, 10))
        assert_allclose(h.Herr, 5 * np.arange(1, 10) ** 0.5)
        assert_allclose(h.A, 5 * np.arange(1, 10))
        assert_allclose(h.Aerr, 5 * np.arange(1, 10) ** 0.5)
        assert_allclose(h.fit(linear, [0.5, 1])[0], [5, 0], atol=1e-6)

        h = Hist1D(range(10), weights=range(10))
        assert_allclose(h.N, np.ones(10))
        assert_allclose(h.H, range(10))
        assert_allclose(h.Herr, range(10))
        assert_allclose(h.A, range(10))
        assert_allclose(h.Aerr, range(10))
        assert_allclose(h.fit(linear, [3, 1])[0], [1, 0], atol=1e-6)

        h = Hist1D(v, weights=v)
        assert_allclose(h.N, range(1, 10))
        assert_allclose(h.H, np.arange(1, 10) ** 2)
        assert_allclose(h.Herr, np.arange(1, 10) ** 1.5)
        assert_allclose(h.A, np.arange(1, 10) ** 2)
        assert_allclose(h.Aerr, np.arange(1, 10) ** 1.5)
        assert_allclose(h.fit(quadratic, [3, 1, 2])[0], [1, 0, 0], atol=1e-6)

        h = Hist1D(v, norm=True)
        assert_allclose(h.N, range(1, 10))
        assert_allclose(h.H, np.arange(1, 10) / 45)
        assert_allclose(h.Herr, (np.arange(1, 10) ** 0.5) / 45)
        assert_allclose(h.A, np.arange(1, 10) / 45)
        assert_allclose(h.Aerr, (np.arange(1, 10) ** 0.5) / 45)
        assert_allclose(h.fit(linear, [1, 2])[0], [1 / 45, 0], atol=1e-6)

        v1 = range(150)
        h = Hist1D(v1)
        self.assertEqual(len(h.N), 50)
        assert_allclose(h.N, 3)
        assert_allclose(h.x.widths, 3)
        assert_allclose(h.A, 3)
        assert_allclose(h.Aerr, 3**0.5)
        assert_allclose(h.H, 1)
        assert_allclose(h.Aerr, 3**0.5)
        assert_allclose(h.fit(linear, [1, 1])[0], [0, 1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
