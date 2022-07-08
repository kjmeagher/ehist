import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ehist.axis import AutoAxis, LinearAxis, LogAxis, print_axis


class TestAxis(unittest.TestCase):
    def test_integer(self):
        a1 = AutoAxis(range(10), 10)
        assert_array_equal(a1.widths, 1)
        assert_array_equal(a1.edges, np.linspace(-0.5, 9.5, 11))
        assert_array_equal(a1.pcenters, range(10))
        assert_array_equal(a1.pedges, a1.edges)

        a2 = AutoAxis(range(10), 20)
        assert_array_equal(a2.widths, 1)
        assert_array_equal(a2.edges, np.linspace(-0.5, 9.5, 11))
        assert_array_equal(a2.pcenters, range(10))
        assert_array_equal(a2.pedges, a2.edges)

        a3 = AutoAxis(range(20), 10)
        assert_array_equal(a3.widths, 2)
        assert_array_equal(a3.edges, range(0, 21, 2))
        assert_array_equal(a3.pcenters, range(1, 21, 2))
        assert_array_equal(a3.pedges, range(0, 21, 2))

        a4 = AutoAxis(range(2, 21), 10)
        assert_array_equal(a4.widths, 9 * [2] + [1])
        assert_array_equal(a4.edges, list(range(2, 21, 2)) + [21])
        assert_array_equal(a4.pcenters, list(range(3, 20, 2)) + [20.5])
        assert_array_equal(a4.pedges, list(range(2, 21, 2)) + [21])

        a5 = AutoAxis(range(3, 22), 10)
        assert_array_equal(a5.widths, [1] + 9 * [2])
        assert_array_equal(a5.edges, [3] + list(range(4, 23, 2)))
        assert_array_equal(a5.pcenters, [3.5] + list(range(5, 22, 2)))
        assert_array_equal(a5.pedges, [3] + list(range(4, 23, 2)))

        a6 = AutoAxis(range(5, 23), 10)
        assert_array_equal(a6.widths, [1] + 8 * [2] + [1])
        assert_array_equal(a6.edges, [5] + list(range(6, 23, 2)) + [23])
        assert_array_equal(a6.pcenters, [5.5] + list(range(7, 22, 2)) + [22.5])
        assert_array_equal(a6.pedges, [5] + list(range(6, 23, 2)) + [23])

        a7 = AutoAxis(range(5, 23), span=(10, 15))
        assert_array_equal(a7.widths, 1)
        assert_array_equal(a7.edges, np.arange(9.5, 16))
        assert_array_equal(a7.pcenters, range(10, 16))
        assert_array_equal(a7.pedges, np.arange(9.5, 16))

        a8 = AutoAxis(range(5, 101), bins=10, span=[11, 90])
        print_axis(a8)
        assert_array_equal(a8.widths, 8)
        assert_array_equal(a8.edges, range(11, 92, 8))
        assert_array_equal(a8.pcenters, range(15, 92, 8))
        assert_array_equal(a8.pedges, range(11, 92, 8))

    def test_LogIntAxis(self):
        b1 = np.arange(1, 11)
        a1 = AutoAxis(b1, 10, t="logint")
        assert_array_equal(a1.bins, b1)
        assert_array_equal(a1.edges, b1 - 0.5)
        assert_array_equal(a1.pedges, b1)
        assert_array_equal(a1.widths, b1[1:] - b1[:-1])
        assert_array_equal(a1.pcenters, (b1[1:] * b1[:-1]) ** 0.5)

        b2 = np.arange(1, 21)
        a2 = AutoAxis(b2, 10, t="logint")
        bb = np.array([1, 2, 3, 4, 5, 7, 9, 12, 15, 20])
        assert_array_equal(a2.bins, bb)
        assert_array_equal(a2.widths, np.diff(bb))
        assert_array_equal(a2.edges, bb - 0.5)
        assert_array_equal(a2.pedges, bb)
        assert_array_equal(a2.pcenters, (bb[1:] * bb[:-1]) ** 0.5)

        a3 = AutoAxis(range(10, 101), 10, t="logint")
        b3 = (np.geomspace(10, 100, 10) + 0.5).astype(int)
        assert_array_equal(a3.bins, b3)
        assert_array_equal(a3.edges, b3 - 0.5)
        assert_array_equal(a3.pedges, b3)
        assert_array_equal(a3.widths, b3[1:] - b3[:-1])
        assert_array_equal(a3.pcenters, (b3[1:] * b3[:-1]) ** 0.5)

        a4 = AutoAxis(range(10, 1000001), 10, t="logint")
        b4 = (np.geomspace(10, 1e6, 10) + 0.5).astype(int)
        assert_allclose(a4.bins, b4, rtol=1e-4)
        assert_allclose(a4.edges, b4 - 0.5, rtol=1e-4)
        assert_allclose(a4.pedges, b4, rtol=1e-4)
        assert_allclose(a4.widths, np.diff(b4), rtol=1e-4)
        assert_allclose(a4.pcenters, (b4[1:] * b4[:-1]) ** 0.5, rtol=1e-4)

        b5 = list(range(1, 11))
        a5 = AutoAxis(b5, bins=b5[::2], t="logint")
        assert_allclose(a5.bins, range(1, 10, 2), rtol=1e-4)
        assert_allclose(a5.edges, np.arange(0.5, 10, 2), rtol=1e-4)
        assert_allclose(a5.pedges, np.arange(1, 10, 2), rtol=1e-4)
        assert_allclose(a5.widths, 2, rtol=1e-4)
        assert_allclose(a5.pcenters, np.sqrt([1 * 3, 3 * 5, 5 * 7, 7 * 9]), rtol=1e-4)
        print_axis(a5)

    def test_linear(self):

        a1 = AutoAxis(np.linspace(0, 10, 100), 10, None, None)
        b1 = np.linspace(0, 10, 11)
        assert_array_equal(a1.bins, b1)
        assert_array_equal(a1.widths, 10 * [1])
        assert_array_equal(a1.edges, b1)
        assert_array_equal(a1.pedges, b1)
        assert_array_equal(a1.pcenters, b1[:-1] + 0.5)

    def test_LogAxis(self):

        b1 = np.geomspace(1, 10, 11)
        a1 = AutoAxis(np.geomspace(1, 10, 100), 10, t="log")
        self.assertTrue(isinstance(a1, LogAxis))
        assert_array_equal(a1.bins, b1)
        assert_array_equal(a1.widths, np.diff(b1))
        assert_array_equal(a1.edges, b1)
        assert_array_equal(a1.pedges, b1)
        assert_array_equal(a1.pcenters, np.sqrt(b1[:-1] * b1[1:]))

    def test_AutoAxis(self):

        with self.assertRaises(ValueError):
            AutoAxis(np.array([], dtype=object))

        with self.assertRaises(ValueError):
            AutoAxis([], t="asdf")

        with self.assertRaises(ValueError):
            AutoAxis([], bins="asdf")

        a1 = AutoAxis(np.geomspace(1, 10, 1000), bins="blocks")
        self.assertTrue(isinstance(a1, LinearAxis))
        self.assertEqual(a1.bins[0], 1)
        self.assertEqual(a1.bins[-1], 10)

        a2 = AutoAxis(np.geomspace(1, 10, 1000), bins="knuth")
        self.assertTrue(isinstance(a2, LinearAxis))
        self.assertEqual(a2.bins[0], 1)
        self.assertEqual(a2.bins[-1], 10)

        a3 = AutoAxis(np.geomspace(1, 10, 1000), bins="scott")
        self.assertTrue(isinstance(a3, LinearAxis))
        self.assertEqual(a3.bins[0], 1)
        self.assertGreaterEqual(a3.bins[-1], 10)

        a3 = AutoAxis(np.geomspace(1, 10, 1000), bins="freedman")
        self.assertTrue(isinstance(a3, LinearAxis))
        self.assertEqual(a3.bins[0], 1)
        self.assertGreaterEqual(a3.bins[-1], 10)


if __name__ == "__main__":
    unittest.main()
