# SPDX-FileCopyrightText: © 2023 The ehist authors
#
# SPDX-License-Identifier: BSD-2-Clause

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Bayesian Blocks for Time Series Analysis
========================================.

Dynamic programming algorithm for solving a piecewise-constant model for
various datasets. This is based on the algorithm presented in Scargle
et al 2012 [1]_. This code was ported from the astroML project [2]_.

Applications include:

- finding an optimal histogram with adaptive bin widths
- finding optimal segmentation of time series data
- detecting inflection points in the rate of event data

The primary interface to these routines is the :func:`bayesian_blocks`
function. This module provides fitness functions suitable for three types
of data:

- Irregularly-spaced event data via the :class:`Events` class
- Regularly-spaced event data via the :class:`RegularEvents` class
- Irregularly-spaced point measurements via the :class:`PointMeasures` class

For more fine-tuned control over the fitness functions used, it is possible
to define custom :class:`FitnessFunc` classes directly and use them with
the :func:`bayesian_blocks` routine.

One common application of the Bayesian Blocks algorithm is the determination
of optimal adaptive-width histogram bins. This uses the same fitness function
as for irregularly-spaced time series events. The easiest interface for
creating Bayesian Blocks histograms is the :func:`astropy.stats.histogram`
function.

References
----------
.. [1] https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S
.. [2] https://www.astroml.org/ https://github.com//astroML/astroML/
"""

import warnings
from inspect import signature

import numpy as np
from scipy import optimize, special

# TODO: implement other fitness functions from appendix B of Scargle 2012  # noqa: FIX002, TD002, TD003

# __all__ = ['FitnessFunc', 'Events', 'RegularEvents', 'PointMeasures',
#           'bayesian_blocks']


def bayesian_blocks(t, x=None, sigma=None, fitness="events", **kwargs):
    r"""Compute optimal segmentation of data with Scargle's Bayesian Blocks.

    This is a flexible implementation of the Bayesian Blocks algorithm
    described in Scargle 2012 [1]_.

    Parameters
    ----------
    t : array_like
        data times (one dimensional, length N)
    x : array_like, optional
        data values
    sigma : array_like or float, optional
        data errors
    fitness : str or object
        the fitness function to use for the model.
        If a string, the following options are supported:

        - 'events' : binned or unbinned event data.  Arguments are ``gamma``,
            which gives the slope of the prior on the number of bins, or
            ``ncp_prior``, which is :math:`-\ln({\tt gamma})`.
        - 'regular_events' : non-overlapping events measured at multiples of a
            fundamental tick rate, ``dt``, which must be specified as an
            additional argument.  Extra arguments are ``p0``, which gives the
            false alarm probability to compute the prior, or ``gamma``, which
            gives the slope of the prior on the number of bins, or ``ncp_prior``,
            which is :math:`-\ln({\tt gamma})`.
        - 'measures' : fitness for a measured sequence with Gaussian errors.
            Extra arguments are ``p0``, which gives the false alarm probability
            to compute the prior, or ``gamma``, which gives the slope of the
            prior on the number of bins, or ``ncp_prior``, which is
            :math:`-\ln({\tt gamma})`.

        In all three cases, if more than one of ``p0``, ``gamma``, and
        ``ncp_prior`` is chosen, ``ncp_prior`` takes precedence over ``gamma``
        which takes precedence over ``p0``.

        Alternatively, the fitness parameter can be an instance of
        :class:`FitnessFunc` or a subclass thereof.

    **kwargs :
        any additional keyword arguments will be passed to the specified
        :class:`FitnessFunc` derived class.

    Returns
    -------
    edges : ndarray
        array containing the (N+1) edges defining the N bins

    Examples
    --------
    .. testsetup::

        >>> np.random.seed(12345)

    Event data:

    >>> t = np.random.normal(size=100)
    >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)

    Event data with repeats:

    >>> t = np.random.normal(size=100)
    >>> t[80:] = t[:20]
    >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)

    Regular event data:

    >>> dt = 0.05
    >>> t = dt * np.arange(1000)
    >>> x = np.zeros(len(t))
    >>> x[np.random.randint(0, len(t), len(t) // 10)] = 1
    >>> edges = bayesian_blocks(t, x, fitness='regular_events', dt=dt)

    Measured point data with errors:

    >>> t = 100 * np.random.random(100)
    >>> x = np.exp(-0.5 * (t - 50) ** 2)
    >>> sigma = 0.1
    >>> x_obs = np.random.normal(x, sigma)
    >>> edges = bayesian_blocks(t, x_obs, sigma, fitness='measures')

    References
    ----------
    .. [1] Scargle, J et al. (2012)
        https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S

    See Also
    --------
    astropy.stats.histogram : compute a histogram using bayesian blocks
    """
    FITNESS_DICT = {
        "events": Events,
        "regular_events": RegularEvents,
        "measures": PointMeasures,
    }
    fitness = FITNESS_DICT.get(fitness, fitness)

    if type(fitness) is type and issubclass(fitness, FitnessFunc):
        fitfunc = fitness(**kwargs)
    elif isinstance(fitness, FitnessFunc):
        fitfunc = fitness
    else:
        raise ValueError("fitness parameter not understood")

    return fitfunc.fit(t, x, sigma)


class FitnessFunc:
    """Base class for bayesian blocks fitness functions.

    Derived classes should overload the following method:

    ``fitness(self, **kwargs)``:
        Compute the fitness given a set of named arguments.
        Arguments accepted by fitness must be among ``[T_k, N_k, a_k, b_k, c_k]``
        (See [1]_ for details on the meaning of these parameters).

    Additionally, other methods may be overloaded as well:

    ``__init__(self, **kwargs)``:
        Initialize the fitness function with any parameters beyond the normal
        ``p0`` and ``gamma``.

    ``validate_input(self, t, x, sigma)``:
        Enable specific checks of the input data (``t``, ``x``, ``sigma``)
        to be performed prior to the fit.

    ``compute_ncp_prior(self, N)``: If ``ncp_prior`` is not defined explicitly,
        this function is called in order to define it before fitting. This may be
        calculated from ``gamma``, ``p0``, or whatever method you choose.

    ``p0_prior(self, N)``:
        Specify the form of the prior given the false-alarm probability ``p0``
        (See [1]_ for details).

    For examples of implemented fitness functions, see :class:`Events`,
    :class:`RegularEvents`, and :class:`PointMeasures`.

    References
    ----------
    .. [1] Scargle, J et al. (2012)
        https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None) -> None:
        self.p0 = p0
        self.gamma = gamma
        self.ncp_prior = ncp_prior

    def validate_input(self, t, x=None, sigma=None):
        """Validate inputs to the model.

        Parameters
        ----------
        t : array_like
            times of observations
        x : array_like, optional
            values observed at each time
        sigma : float or array_like, optional
            errors in values x

        Returns
        -------
        t, x, sigma : array_like, float or None
            validated and perhaps modified versions of inputs
        """
        # validate array input
        t = np.asarray(t, dtype=float)

        # find unique values of t
        t = np.array(t)
        if t.ndim != 1:
            raise ValueError("t must be a one-dimensional array")
        unq_t, unq_ind, unq_inv = np.unique(t, return_index=True, return_inverse=True)

        # if x is not specified, x will be counts at each time
        if x is None:
            if sigma is not None:
                raise ValueError("If sigma is specified, x must be specified")
            else:
                sigma = 1

            x = np.ones_like(t) if len(unq_t) == len(t) else np.bincount(unq_inv)

            t = unq_t

        # if x is specified, then we need to simultaneously sort t and x
        else:
            # TODO: allow broadcasted x?  # noqa: FIX002, TD002, TD003
            x = np.asarray(x, dtype=float)

            if x.shape not in [(), (1,), (t.size,)]:
                raise ValueError("x does not match shape of t")
            x += np.zeros_like(t)

            if len(unq_t) != len(t):
                raise ValueError("Repeated values in t not supported when x is specified")
            t = unq_t
            x = x[unq_ind]

        # verify the given sigma value
        if sigma is None:
            sigma = 1
        else:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.shape not in [(), (1,), (t.size,)]:
                raise ValueError("sigma does not match the shape of x")

        return t, x, sigma

    def fitness(self, **kwargs):
        raise NotImplementedError

    def p0_prior(self, N):
        """Empirical prior, parametrized by the false alarm probability ``p0``
        See  eq. 21 in Scargle (2012).

        Note that there was an error in this equation in the original Scargle
        paper (the "log" was missing). The following corrected form is taken
        from https://arxiv.org/abs/1304.2818
        """
        return 4 - np.log(73.53 * self.p0 * (N**-0.478))

    # the fitness_args property will return the list of arguments accepted by
    # the method fitness().  This allows more efficient computation below.
    @property
    def _fitness_args(self):
        return signature(self.fitness).parameters.keys()

    def compute_ncp_prior(self, N):
        """If ``ncp_prior`` is not explicitly defined, compute it from ``gamma``
        or ``p0``.
        """
        if self.gamma is not None:
            return -np.log(self.gamma)
        elif self.p0 is not None:
            return self.p0_prior(N)
        else:
            raise ValueError(
                "``ncp_prior`` cannot be computed as neither ``gamma`` nor ``p0`` is defined.",
            )

    def fit(self, t, x=None, sigma=None):
        """Fit the Bayesian Blocks model given the specified fitness function.

        Parameters
        ----------
        t : array_like
            data times (one dimensional, length N)
        x : array_like, optional
            data values
        sigma : array_like or float, optional
            data errors

        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """
        t, x, sigma = self.validate_input(t, x, sigma)

        # compute values needed for computation, below
        if "a_k" in self._fitness_args:
            ak_raw = np.ones_like(x) / sigma**2
        if "b_k" in self._fitness_args:
            bk_raw = x / sigma**2
        if "c_k" in self._fitness_args:
            ck_raw = x * x / sigma**2

        # create length-(N + 1) array of cell edges
        edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
        block_length = t[-1] - edges

        # arrays to store the best configuration
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)

        # Compute ncp_prior if not defined
        ncp_prior = self.compute_ncp_prior(N) if self.ncp_prior is None else self.ncp_prior

        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------
        for R in range(N):
            # Compute fit_vec : fitness of putative last block (end at R)
            kwds = {}

            # T_k: width/duration of each block
            if "T_k" in self._fitness_args:
                kwds["T_k"] = block_length[: R + 1] - block_length[R + 1]

            # N_k: number of elements in each block
            if "N_k" in self._fitness_args:
                kwds["N_k"] = np.cumsum(x[: R + 1][::-1])[::-1]

            # a_k: eq. 31
            if "a_k" in self._fitness_args:
                kwds["a_k"] = 0.5 * np.cumsum(ak_raw[: R + 1][::-1])[::-1]

            # b_k: eq. 32
            if "b_k" in self._fitness_args:
                kwds["b_k"] = -np.cumsum(bk_raw[: R + 1][::-1])[::-1]

            # c_k: eq. 33
            if "c_k" in self._fitness_args:
                kwds["c_k"] = 0.5 * np.cumsum(ck_raw[: R + 1][::-1])[::-1]

            # evaluate fitness function
            fit_vec = self.fitness(**kwds)

            A_R = fit_vec - ncp_prior
            A_R[1:] += best[:R]

            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]

        # ----------------------------------------------------------------
        # Now find changepoints by iteratively peeling off the last block
        # ----------------------------------------------------------------
        change_points = np.zeros(N, dtype=int)
        i_cp = N
        ind = N
        while i_cp > 0:
            i_cp -= 1
            change_points[i_cp] = ind
            if ind == 0:
                break
            ind = last[ind - 1]
        if i_cp == 0:
            change_points[i_cp] = 0
        change_points = change_points[i_cp:]

        return edges[change_points]


class Events(FitnessFunc):
    r"""Bayesian blocks fitness for binned or unbinned events.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on
        :math:`N_{\rm blocks}` (see eq. 21 of Scargle 2012). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this fitness function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \sim {\tt gamma}^{N_{\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """

    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2012
        return N_k * (np.log(N_k) - np.log(T_k))

    def validate_input(self, t, x, sigma):
        t, x, sigma = super().validate_input(t, x, sigma)
        if x is not None and np.any(x % 1 > 0):
            raise ValueError("x must be integer counts for fitness='events'")
        return t, x, sigma


class RegularEvents(FitnessFunc):
    r"""Bayesian blocks fitness for regular events.

    This is for data which has a fundamental "tick" length, so that all
    measured values are multiples of this tick length.  In each tick, there
    are either zero or one counts.

    Parameters
    ----------
    dt : float
        tick rate for data
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\rm
        blocks}` (see eq. 21 of Scargle 2012). If gamma is specified, p0 is
        ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """

    def __init__(self, dt, p0=0.05, gamma=None, ncp_prior=None) -> None:
        self.dt = dt
        super().__init__(p0, gamma, ncp_prior)

    def validate_input(self, t, x, sigma):
        t, x, sigma = super().validate_input(t, x, sigma)
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("Regular events must have only 0 and 1 in x")
        return t, x, sigma

    def fitness(self, T_k, N_k):
        # Eq. 75 of Scargle 2012
        M_k = T_k / self.dt
        N_over_M = N_k / M_k

        eps = 1e-8
        if np.any(N_over_M > 1 + eps):
            warnings.warn("regular events: N/M > 1.  Is the time step correct?")

        one_m_NM = 1 - N_over_M
        N_over_M[N_over_M <= 0] = 1
        one_m_NM[one_m_NM <= 0] = 1

        return N_k * np.log(N_over_M) + (M_k - N_k) * np.log(one_m_NM)


class PointMeasures(FitnessFunc):
    r"""Bayesian blocks fitness for point measures.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\rm
        blocks}` (see eq. 21 of Scargle 2012). If gamma is specified, p0 is
        ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None) -> None:
        super().__init__(p0, gamma, ncp_prior)

    def fitness(self, a_k, b_k):
        # eq. 41 from Scargle 2012
        return (b_k * b_k) / (4 * a_k)

    def validate_input(self, t, x, sigma):
        if x is None:
            raise ValueError("x must be specified for point measures")
        return super().validate_input(t, x, sigma)


def scott_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using Scott's rule
    Scott's rule is a normal reference rule: it minimizes the integrated
    mean squared error in the bin approximation under the assumption that the
    data is approximately Gaussian.

    Parameters
    ----------
    data : array_like, ndim=1
        observed (one-dimensional) data
    return_bins : bool, optional
        if True, then return the bin edges
    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if ``return_bins`` is True
    Notes
    -----
    The optimal bin width is
    .. math::
        \Delta_b = \frac{3.5\sigma}{n^{1/3}}
    where :math:`\sigma` is the standard deviation of the data, and
    :math:`n` is the number of data points [1]_.


    References
    ----------
    .. [1] Scott, David W. (1979). "On optimal and data-based histograms".
        Biometricka 66 (3): 605-610
    See Also
    --------
    knuth_bin_width
    freedman_bin_width
    bayesian_blocks
    histogram.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    sigma = np.std(data)

    dx = 3.5 * sigma / (n ** (1 / 3))

    if return_bins:
        Nbins = np.ceil((data.max() - data.min()) / dx)
        Nbins = max(1, Nbins)
        bins = data.min() + dx * np.arange(Nbins + 1)
        return dx, bins
    else:
        return dx


def freedman_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using the Freedman-Diaconis rule
    The Freedman-Diaconis rule is a normal reference rule like Scott's
    rule, but uses rank-based statistics for results which are more robust
    to deviations from a normal distribution.

    Parameters
    ----------
    data : array_like, ndim=1
        observed (one-dimensional) data
    return_bins : bool, optional
        if True, then return the bin edges
    Returns
    -------
    width : float
        optimal bin width using the Freedman-Diaconis rule
    bins : ndarray
        bin edges: returned if ``return_bins`` is True
    Notes
    -----
    The optimal bin width is
    .. math::
        \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}
    where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
    :math:`n` is the number of data points [1]_.


    References
    ----------
    .. [1] D. Freedman & P. Diaconis (1981)
        "On the histogram as a density estimator: L2 theory".
        Probability Theory and Related Fields 57 (4): 453-476
    See Also
    --------
    knuth_bin_width
    scott_bin_width
    bayesian_blocks
    histogram.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    v25, v75 = np.percentile(data, [25, 75])
    dx = 2 * (v75 - v25) / (n ** (1 / 3))

    if return_bins:
        dmin, dmax = data.min(), data.max()
        Nbins = max(1, np.ceil((dmax - dmin) / dx))
        try:
            bins = dmin + dx * np.arange(Nbins + 1)
        except ValueError as e:
            if "Maximum allowed size exceeded" in str(e):
                raise ValueError(
                    "The inter-quartile range of the data is too small: "
                    f"failed to construct histogram with {Nbins + 1} bins. "
                    "Please use another bin method, such as "
                    'bins="scott"',
                )
            else:  # Something else  # pragma: no cover
                raise
        return dx, bins
    else:
        return dx


def knuth_bin_width(data, return_bins=False, quiet=True):
    r"""Return the optimal histogram bin width using Knuth's rule.
    Knuth's rule is a fixed-width, Bayesian approach to determining
    the optimal bin width of a histogram.

    Parameters
    ----------
    data : array_like, ndim=1
        observed (one-dimensional) data
    return_bins : bool, optional
        if True, then return the bin edges
    quiet : bool, optional
        if True (default) then suppress stdout output from scipy.optimize
    Returns
    -------
    dx : float
        optimal bin width. Bins are measured starting at the first data point.
    bins : ndarray
        bin edges: returned if ``return_bins`` is True
    Notes
    -----
    The optimal number of bins is the value M which maximizes the function
    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})
    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`
    [1]_.


    References
    ----------
    .. [1] Knuth, K.H. "Optimal Data-Based Binning for Histograms".
        arXiv:0605197, 2006
    See Also
    --------
    freedman_bin_width
    scott_bin_width
    bayesian_blocks
    histogram.
    """
    # import here because of optional scipy dependency

    knuthF = _KnuthF(data)
    dx0, bins0 = freedman_bin_width(data, True)
    M = optimize.fmin(knuthF, len(bins0), disp=not quiet)[0]
    bins = knuthF.bins(M)
    dx = bins[1] - bins[0]

    if return_bins:
        return dx, bins
    else:
        return dx


class _KnuthF:
    r"""Class which implements the function minimized by knuth_bin_width
    Parameters
    ----------
    data : array_like, one dimension
        data to be histogrammed
    Notes
    -----
    the function F is given by
    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})
    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`.

    See Also
    --------
    knuth_bin_width.
    """

    def __init__(self, data) -> None:
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size

        # create a reference to gammaln to use in self.eval()
        self.gammaln = special.gammaln

    def bins(self, M):
        """Return the bin edges given a width dx."""
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        """Evaluate the Knuth function
        Parameters
        ----------
        dx : float
            Width of bins
        Returns
        -------
        F : float
            evaluation of the negative Knuth likelihood function:
            smaller values indicate a better fit.
        """
        M = int(M)

        if M <= 0:
            return np.inf

        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(
            self.n * np.log(M)
            + self.gammaln(0.5 * M)
            - M * self.gammaln(0.5)
            - self.gammaln(self.n + 0.5 * M)
            + np.sum(self.gammaln(nk + 0.5))
        )
