import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ehist.axis import AutoAxis


class TestAxis(unittest.TestCase):
    def test_integer(self):
        a1 = AutoAxis(range(10), 10, None, None)
        assert_array_equal(a1.widths, 1)
        assert_array_equal(a1.edges, np.linspace(-0.5, 9.5, 11))
        assert_array_equal(a1.pcenters, range(10))
        assert_array_equal(a1.pedges, a1.edges)

        a2 = AutoAxis(range(10), 20, None, None)
        assert_array_equal(a2.widths, 1)
        assert_array_equal(a2.edges, np.linspace(-0.5, 9.5, 11))
        assert_array_equal(a2.pcenters, range(10))
        assert_array_equal(a2.pedges, a2.edges)

        a3 = AutoAxis(range(20), 10, None, None)
        assert_array_equal(a3.widths, 2)
        assert_array_equal(a3.edges, range(0, 21, 2))
        assert_array_equal(a3.pcenters, range(1, 21, 2))
        assert_array_equal(a3.pedges, range(0, 21, 2))

        a4 = AutoAxis(range(2, 21), 10, None, None)
        assert_array_equal(a4.widths, 9 * [2] + [1])
        assert_array_equal(a4.edges, list(range(2, 21, 2)) + [21])
        assert_array_equal(a4.pcenters, list(range(3, 20, 2)) + [20.5])
        assert_array_equal(a4.pedges, list(range(2, 21, 2)) + [21])

        a5 = AutoAxis(range(3, 22), 10, None, None)
        assert_array_equal(a5.widths, [1] + 9 * [2])
        assert_array_equal(a5.edges, [3] + list(range(4, 23, 2)))
        assert_array_equal(a5.pcenters, [3.5] + list(range(5, 22, 2)))
        assert_array_equal(a5.pedges, [3] + list(range(4, 23, 2)))

        a6 = AutoAxis(range(5, 23), 10, None, None)
        assert_array_equal(a6.widths, [1] + 8 * [2] + [1])
        assert_array_equal(a6.edges, [5] + list(range(6, 23, 2)) + [23])
        assert_array_equal(a6.pcenters, [5.5] + list(range(7, 22, 2)) + [22.5])
        assert_array_equal(a6.pedges, [5] + list(range(6, 23, 2)) + [23])

    def test_LogIntAxis(self):
        b1 = np.arange(1, 11)
        a1 = AutoAxis(b1, 10, None, "logint")
        assert_array_equal(a1.bins, b1)
        assert_array_equal(a1.edges, b1 - 0.5)
        assert_array_equal(a1.pedges, b1)
        assert_array_equal(a1.widths, b1[1:] - b1[:-1])
        assert_array_equal(a1.pcenters, (b1[1:] * b1[:-1]) ** 0.5)

        b2 = np.arange(1, 21)
        a2 = AutoAxis(b2, 10, None, "logint")
        bb = np.array([1, 2, 3, 4, 5, 7, 9, 12, 15, 20])
        assert_array_equal(a2.bins, bb)
        assert_array_equal(a2.widths, np.diff(bb))
        assert_array_equal(a2.edges, bb - 0.5)
        assert_array_equal(a2.pedges, bb)
        assert_array_equal(a2.pcenters, (bb[1:] * bb[:-1]) ** 0.5)

        a3 = AutoAxis(range(10, 101), 10, None, "logint")
        b3 = (np.geomspace(10, 100, 10) + 0.5).astype(int)
        assert_array_equal(a3.bins, b3)
        assert_array_equal(a3.edges, b3 - 0.5)
        assert_array_equal(a3.pedges, b3)
        assert_array_equal(a3.widths, b3[1:] - b3[:-1])
        assert_array_equal(a3.pcenters, (b3[1:] * b3[:-1]) ** 0.5)

        a4 = AutoAxis(range(10, 1000001), 10, None, "logint")
        b4 = (np.geomspace(10, 1e6, 10) + 0.5).astype(int)
        assert_allclose(a4.bins, b4, rtol=1e-4)
        assert_allclose(a4.edges, b4 - 0.5, rtol=1e-4)
        assert_allclose(a4.pedges, b4, rtol=1e-4)
        assert_allclose(a4.widths, np.diff(b4), rtol=1e-4)
        assert_allclose(a4.pcenters, (b4[1:] * b4[:-1]) ** 0.5, rtol=1e-4)

    def test_linear(self):

        a1 = AutoAxis(np.linspace(0, 10, 100), 10, None, None)
        b1 = np.linspace(0, 10, 11)
        assert_array_equal(a1.bins, b1)
        assert_array_equal(a1.widths, 10 * [1])
        assert_array_equal(a1.edges, b1)
        assert_array_equal(a1.pedges, b1)
        assert_array_equal(a1.pcenters, b1[:-1] + 0.5)


if __name__ == "__main__":
    unittest.main()
