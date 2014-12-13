#!/usr/bin/env python
# coding: utf-8
"""
Audio Filters
"""

# Python 2.7 Standard Library
import abc
import doctest
import sys

# Third-Party Libraries
import numpy as np

# Local Library
from audio.fourier import F

#
# Metadata
# ------------------------------------------------------------------------------
#
from .about_filters import *

#
# TODO
# ------------------------------------------------------------------------------
#   - refactor: setting a at start in filters mandatory ? And after that, you
#     cannot change the size of a ? Or you can set a from None only ONCE and
#     after that the size is fixed ? Dunno. Is it really simpler ?
#   - add FIR low-pass (or band-pass) filters. Solve PROPERLY the even_number
#     issue (upsample with low-pass content assumption ?)
#   - check AR API wrt leading coeff. General enough ?
#   - support transfer function for FIR and AR ?
#   - stability test for AR ? (answers True for FIR :))
#   - spectra ... Mmmm spectrum is maybe not the right word ... 
#     "frequency response would be better ... still ...
#   - put filter banks here ? submodule QMF ?
#   - is the ability to change the size of the filter an anti-pattern ?
#dependencies: script, spectrum

#  - kill script dependency,

#  - reconsider the support for `F` w.r.t. evolutions of `audio.fourier`.

#Core:

#  - make filter length fixed (dynamic length is too smart).

#  - consider read-only filter parameters (tricky: arrays can be changed but
#    not replaced, won't work for scalar params that shall be mutable).

#  - use opaque filter states (get/set semantics).

#  - support lattice filters (for linear prediction).

#  - add gain parameter to `AR` filter (?).

#  - generic parameter interface for all filters (tricky).

#  - cythonize for performance & correctness.

#Extra:

#  - `low_pass`: ???

#  - `MPEG` stuff & filter banks: ???

#
# Filter Abstract Base Class
# ------------------------------------------------------------------------------
#

class Filter(object):
    __metaclass__ = abc.ABCMeta
    """
    Filters Base Class.
    """
    @abc.abstractmethod
    def __init__(self, a):
        """
        Argument
        --------

          - `a`: the filter coefficients, a sequence of numbers,
        """

        self._a = np.array(a, ndmin=1)
        
    def get_a(self):
        """
        Return a view of the filter coefficients.
        """
        return self._a
            
    a = property(get_a)

    # TODO: document the legit use of state: as an opaque variable only.

    def get_state(self):
        return np.copy(self._state)

    def set_state(self, state=None):
        if state is None:
            self._state = np.zeros_like(self._state)
        else:
            self._state[:] = state

    state = property(get_state, set_state) 

    @abc.abstractmethod
    def __call__(self, input):
        """
        Compute new filter output value(s)

        Argument
        --------

          - `input`: number or sequence of numbers

        Returns
        -------

          - `output`: number or sequence of numbers
        """
        pass

    @abc.abstractmethod
    def poles(self):
        """
        Compute the filter poles.
        """
        pass

    @abc.abstractmethod
    def __F__(self, *kwargs):
        """
        Return the filter frequency response.
        """
        pass

    def __repr__(self):
        type_name = type(self).__name__
        args = (type_name, list(self.a))
        return "{0}(a={1})".format(*args)

    __str__ = __repr__


#
# Finite Impulse Response Filter
# ------------------------------------------------------------------------------
#


class FIR(Filter):
    """
Finite Impulse Response Filter

    y[n] = a[0] * u[n] + ... + a[N-1] * u[n-N+1]
"""   

    def __init__(self, a):
        Filter.__init__(self, a)
        self._state = np.zeros_like(self.a[1:])

    def __call__(self, input):
        if np.shape(input):
            inputs = np.ravel(input)
            return np.array([self(input) for input in inputs])
        else:
            # The state array stores the most recent values first.
            output = self._a[0] * input + np.dot(self._a[1:], self.state)
            if len(self._state):
                self._state = np.r_[input, self.state[:-1]]
            return output

    def poles(self):
        return np.zeros(len(self.a))

    def __F__(self, *args, **kwargs): 
        # TODO: support other arguments beyond dt ? 
        # TODO: make it work when dt is positional.
        dt = kwargs.get("dt") or 1.0
        return F(self.a / dt, dt=dt) 

#
# Auto-Regressive Filter
# ------------------------------------------------------------------------------
#

class AR(Filter):
    """
Auto-Regressive Filter

    y[n] = a[0] * y[n-1] + ... + a[N-1] * y[n-N+1] + u[n]
    """

    def __init__(self, a):
        Filter.__init__(self, a)
        self._state = np.zeros_like(self.a)

    def __call__(self, input):
        if np.isscalar(input):
            output = np.dot(self.a, self._state) + input
            self._state[1:] = self._state[:-1]
            self._state[0] = output
            return output
        else:
            input = np.array(input)
            a = self._a
            output = np.zeros_like(input)
            for i, _input in enumerate(input):
                _output = np.dot(a, self._state) + _input
                self._state[1:] = self._state[:-1]
                self._state[0] = _output
                output[i] = _output 
            return output

    def poles(self):
        return np.roots(r_[1.0, -self.a])

    def __F__(self, *args, **kwargs):
        # TODO: support other arguments beyond dt ? 
        # TODO: make it work when dt is positional.
        dt = kwargs.get("dt") or 1.0
        FIR_spectrum = F(FIR(a=r_[1.0, -self.a]), dt=dt)
        def AR_spectrum(f):
            return 1.0 / FIR_spectrum(f)
        return AR_spectrum

#
# ------------------------------------------------------------------------------
#
# Why not return as an FIR instance ? (h = FIR.a, spectrum implemented, etc.)
# Dunno. Keep the 'convol-based stuff apart from the "real-time" implementation ?

def low_pass(fc, dt=1.0, window=np.ones):
    if not 0 <= fc <= 0.5 / dt:
        template  = "invalid cutoff frequency fc={0}: "
        template += "0 <= fc <= 0.5/dt = {1}Hz does not hold."
        message = template.format(fc, 0.5 / dt)
        raise ValueError(message)
    def h(n):
        t = np.arange(-0.5 * (n-1), 0.5 * (n-1) + 1) * dt
        return 2 * fc * np.sinc(2 * fc * t) * window(n)
    return h

#
# Filter Banks and Distorsion
# ------------------------------------------------------------------------------
#
class MPEG(object):
    """MPEG-1/2 Audio Layer I+II PQMF Filter Banks

    Attributes
    ----------
      - `A`: analysis filter bank, a `(M, N)` numpy array of floats.
        
      - `S`: synthesis filter bank, a `(M, N)` numpy array of floats.

      - `h0`: PQMF prototype filter coefficients (cutoff frequency `fc = df/M/2`). 

        A 1d numpy array whose length is `513`, with `h0[0] = h0[N-1] = 0.0`. 

      - `df`: sampling frequency (`44100.0`),
  
      - `dt`: sample time (`1/df`),

      - `M`: number of filters in the banks (`32`),

      - `N`: filters length (`512`).
"""

    df = 44100.0
    dt = 1.0 / df
    M = 32
    N = 513

    h0 = np.array(
[  0.00000000e+00,  -4.77000000e-07,  -4.77000000e-07,  -4.77000000e-07,
  -4.77000000e-07,  -4.77000000e-07,  -4.77000000e-07,  -9.54000000e-07,
  -9.54000000e-07,  -9.54000000e-07,  -9.54000000e-07,  -1.43100000e-06,
  -1.43100000e-06,  -1.90700000e-06,  -1.90700000e-06,  -2.38400000e-06,
  -2.38400000e-06,  -2.86100000e-06,  -3.33800000e-06,  -3.33800000e-06,
  -3.81500000e-06,  -4.29200000e-06,  -4.76800000e-06,  -5.24500000e-06,
  -6.19900000e-06,  -6.67600000e-06,  -7.62900000e-06,  -8.10600000e-06,
  -9.06000000e-06,  -1.00140000e-05,  -1.14440000e-05,  -1.23980000e-05,
  -1.38280000e-05,  -1.47820000e-05,  -1.66890000e-05,  -1.81200000e-05,
  -1.95500000e-05,  -2.14580000e-05,  -2.33650000e-05,  -2.52720000e-05,
  -2.76570000e-05,  -3.00410000e-05,  -3.24250000e-05,  -3.48090000e-05,
  -3.76700000e-05,  -4.05310000e-05,  -4.33920000e-05,  -4.62530000e-05,
  -4.95910000e-05,  -5.29290000e-05,  -5.57900000e-05,  -5.96050000e-05,
  -6.29430000e-05,  -6.62800000e-05,  -7.00950000e-05,  -7.34330000e-05,
  -7.67710000e-05,  -8.05850000e-05,  -8.39230000e-05,  -8.72610000e-05,
  -9.05990000e-05,  -9.34600000e-05,  -9.63210000e-05,  -9.91820000e-05,
  -1.01566000e-04,  -1.03951000e-04,  -1.05858000e-04,  -1.07288000e-04,
  -1.08242000e-04,  -1.08719000e-04,  -1.08719000e-04,  -1.08242000e-04,
  -1.06812000e-04,  -1.05381000e-04,  -1.02520000e-04,  -9.91820000e-05,
  -9.53670000e-05,  -9.01220000e-05,  -8.44000000e-05,  -7.77240000e-05,
  -6.96180000e-05,  -6.05580000e-05,  -5.05450000e-05,  -3.95770000e-05,
  -2.71800000e-05,  -1.38280000e-05,   9.54000000e-07,   1.71660000e-05,
   3.43320000e-05,   5.29290000e-05,   7.29560000e-05,   9.39370000e-05,
   1.16348000e-04,   1.40190000e-04,   1.65462000e-04,   1.91212000e-04,
   2.18868000e-04,   2.47478000e-04,   2.77042000e-04,   3.07560000e-04,
   3.39031000e-04,   3.71456000e-04,   4.04358000e-04,   4.38213000e-04,
   4.72546000e-04,   5.07355000e-04,   5.42164000e-04,   5.76973000e-04,
   6.11782000e-04,   6.46591000e-04,   6.80923000e-04,   7.14302000e-04,
   7.47204000e-04,   7.79152000e-04,   8.09669000e-04,   8.38757000e-04,
   8.66413000e-04,   8.91685000e-04,   9.15051000e-04,   9.35555000e-04,
   9.54151000e-04,   9.68933000e-04,   9.80854000e-04,   9.89437000e-04,
   9.94205000e-04,   9.95159000e-04,   9.91821000e-04,   9.83715000e-04,
   9.71317000e-04,   9.53674000e-04,   9.30786000e-04,   9.02653000e-04,
   8.68797000e-04,   8.29220000e-04,   7.83920000e-04,   7.31945000e-04,
   6.74248000e-04,   6.10352000e-04,   5.39303000e-04,   4.62532000e-04,
   3.78609000e-04,   2.88486000e-04,   1.91689000e-04,   8.82150000e-05,
  -2.14580000e-05,  -1.37329000e-04,  -2.59876000e-04,  -3.88145000e-04,
  -5.22137000e-04,  -6.61850000e-04,  -8.06808000e-04,  -9.56535000e-04,
  -1.11103100e-03,  -1.26981700e-03,  -1.43241900e-03,  -1.59788100e-03,
  -1.76668200e-03,  -1.93738900e-03,  -2.11000400e-03,  -2.28309600e-03,
  -2.45714200e-03,  -2.63071100e-03,  -2.80332600e-03,  -2.97403300e-03,
  -3.14188000e-03,  -3.30686600e-03,  -3.46708300e-03,  -3.62253200e-03,
  -3.77178200e-03,  -3.91435600e-03,  -4.04882400e-03,  -4.17470900e-03,
  -4.29058100e-03,  -4.39596200e-03,  -4.48989900e-03,  -4.57048400e-03,
  -4.63819500e-03,  -4.69112400e-03,  -4.72831700e-03,  -4.74882100e-03,
  -4.75215900e-03,  -4.73737700e-03,  -4.70304500e-03,  -4.64916200e-03,
  -4.57382200e-03,  -4.47702400e-03,  -4.35781500e-03,  -4.21524000e-03,
  -4.04930100e-03,  -3.85856600e-03,  -3.64303600e-03,  -3.40175600e-03,
  -3.13472700e-03,  -2.84147300e-03,  -2.52151500e-03,  -2.17485400e-03,
  -1.80053700e-03,  -1.39951700e-03,  -9.71317000e-04,  -5.15938000e-04,
  -3.33790000e-05,   4.75883000e-04,   1.01184800e-03,   1.57356300e-03,
   2.16150300e-03,   2.77423900e-03,   3.41129300e-03,   4.07218900e-03,
   4.75645100e-03,   5.46217000e-03,   6.18934600e-03,   6.93702700e-03,
   7.70330400e-03,   8.48722500e-03,   9.28783400e-03,   1.01037030e-02,
   1.09333990e-02,   1.17750170e-02,   1.26276020e-02,   1.34892460e-02,
   1.43585210e-02,   1.52335170e-02,   1.61128040e-02,   1.69944760e-02,
   1.78761480e-02,   1.87568660e-02,   1.96342470e-02,   2.05068590e-02,
   2.13723180e-02,   2.22287180e-02,   2.30741500e-02,   2.39071850e-02,
   2.47254370e-02,   2.55270000e-02,   2.63109210e-02,   2.70738600e-02,
   2.78153420e-02,   2.85329820e-02,   2.92248730e-02,   2.98900600e-02,
   3.05266380e-02,   3.11326980e-02,   3.17068100e-02,   3.22480200e-02,
   3.27548980e-02,   3.32255360e-02,   3.36599350e-02,   3.40557100e-02,
   3.44128610e-02,   3.47304340e-02,   3.50070000e-02,   3.52420810e-02,
   3.54352000e-02,   3.55863570e-02,   3.56941220e-02,   3.57589720e-02,
   3.57809070e-02,   3.57589720e-02,   3.56941220e-02,   3.55863570e-02,
   3.54352000e-02,   3.52420810e-02,   3.50070000e-02,   3.47304340e-02,
   3.44128610e-02,   3.40557100e-02,   3.36599350e-02,   3.32255360e-02,
   3.27548980e-02,   3.22480200e-02,   3.17068100e-02,   3.11326980e-02,
   3.05266380e-02,   2.98900600e-02,   2.92248730e-02,   2.85329820e-02,
   2.78153420e-02,   2.70738600e-02,   2.63109210e-02,   2.55270000e-02,
   2.47254370e-02,   2.39071850e-02,   2.30741500e-02,   2.22287180e-02,
   2.13723180e-02,   2.05068590e-02,   1.96342470e-02,   1.87568660e-02,
   1.78761480e-02,   1.69944760e-02,   1.61128040e-02,   1.52335170e-02,
   1.43585210e-02,   1.34892460e-02,   1.26276020e-02,   1.17750170e-02,
   1.09333990e-02,   1.01037030e-02,   9.28783400e-03,   8.48722500e-03,
   7.70330400e-03,   6.93702700e-03,   6.18934600e-03,   5.46217000e-03,
   4.75645100e-03,   4.07218900e-03,   3.41129300e-03,   2.77423900e-03,
   2.16150300e-03,   1.57356300e-03,   1.01184800e-03,   4.75883000e-04,
  -3.33790000e-05,  -5.15938000e-04,  -9.71317000e-04,  -1.39951700e-03,
  -1.80053700e-03,  -2.17485400e-03,  -2.52151500e-03,  -2.84147300e-03,
  -3.13472700e-03,  -3.40175600e-03,  -3.64303600e-03,  -3.85856600e-03,
  -4.04930100e-03,  -4.21524000e-03,  -4.35781500e-03,  -4.47702400e-03,
  -4.57382200e-03,  -4.64916200e-03,  -4.70304500e-03,  -4.73737700e-03,
  -4.75215900e-03,  -4.74882100e-03,  -4.72831700e-03,  -4.69112400e-03,
  -4.63819500e-03,  -4.57048400e-03,  -4.48989900e-03,  -4.39596200e-03,
  -4.29058100e-03,  -4.17470900e-03,  -4.04882400e-03,  -3.91435600e-03,
  -3.77178200e-03,  -3.62253200e-03,  -3.46708300e-03,  -3.30686600e-03,
  -3.14188000e-03,  -2.97403300e-03,  -2.80332600e-03,  -2.63071100e-03,
  -2.45714200e-03,  -2.28309600e-03,  -2.11000400e-03,  -1.93738900e-03,
  -1.76668200e-03,  -1.59788100e-03,  -1.43241900e-03,  -1.26981700e-03,
  -1.11103100e-03,  -9.56535000e-04,  -8.06808000e-04,  -6.61850000e-04,
  -5.22137000e-04,  -3.88145000e-04,  -2.59876000e-04,  -1.37329000e-04,
  -2.14580000e-05,   8.82150000e-05,   1.91689000e-04,   2.88486000e-04,
   3.78609000e-04,   4.62532000e-04,   5.39303000e-04,   6.10352000e-04,
   6.74248000e-04,   7.31945000e-04,   7.83920000e-04,   8.29220000e-04,
   8.68797000e-04,   9.02653000e-04,   9.30786000e-04,   9.53674000e-04,
   9.71317000e-04,   9.83715000e-04,   9.91821000e-04,   9.95159000e-04,
   9.94205000e-04,   9.89437000e-04,   9.80854000e-04,   9.68933000e-04,
   9.54151000e-04,   9.35555000e-04,   9.15051000e-04,   8.91685000e-04,
   8.66413000e-04,   8.38757000e-04,   8.09669000e-04,   7.79152000e-04,
   7.47204000e-04,   7.14302000e-04,   6.80923000e-04,   6.46591000e-04,
   6.11782000e-04,   5.76973000e-04,   5.42164000e-04,   5.07355000e-04,
   4.72546000e-04,   4.38213000e-04,   4.04358000e-04,   3.71456000e-04,
   3.39031000e-04,   3.07560000e-04,   2.77042000e-04,   2.47478000e-04,
   2.18868000e-04,   1.91212000e-04,   1.65462000e-04,   1.40190000e-04,
   1.16348000e-04,   9.39370000e-05,   7.29560000e-05,   5.29290000e-05,
   3.43320000e-05,   1.71660000e-05,   9.54000000e-07,  -1.38280000e-05,
  -2.71800000e-05,  -3.95770000e-05,  -5.05450000e-05,  -6.05580000e-05,
  -6.96180000e-05,  -7.77240000e-05,  -8.44000000e-05,  -9.01220000e-05,
  -9.53670000e-05,  -9.91820000e-05,  -1.02520000e-04,  -1.05381000e-04,
  -1.06812000e-04,  -1.08242000e-04,  -1.08719000e-04,  -1.08719000e-04,
  -1.08242000e-04,  -1.07288000e-04,  -1.05858000e-04,  -1.03951000e-04,
  -1.01566000e-04,  -9.91820000e-05,  -9.63210000e-05,  -9.34600000e-05,
  -9.05990000e-05,  -8.72610000e-05,  -8.39230000e-05,  -8.05850000e-05,
  -7.67710000e-05,  -7.34330000e-05,  -7.00950000e-05,  -6.62800000e-05,
  -6.29430000e-05,  -5.96050000e-05,  -5.57900000e-05,  -5.29290000e-05,
  -4.95910000e-05,  -4.62530000e-05,  -4.33920000e-05,  -4.05310000e-05,
  -3.76700000e-05,  -3.48090000e-05,  -3.24250000e-05,  -3.00410000e-05,
  -2.76570000e-05,  -2.52720000e-05,  -2.33650000e-05,  -2.14580000e-05,
  -1.95500000e-05,  -1.81200000e-05,  -1.66890000e-05,  -1.47820000e-05,
  -1.38280000e-05,  -1.23980000e-05,  -1.14440000e-05,  -1.00140000e-05,
  -9.06000000e-06,  -8.10600000e-06,  -7.62900000e-06,  -6.67600000e-06,
  -6.19900000e-06,  -5.24500000e-06,  -4.76800000e-06,  -4.29200000e-06,
  -3.81500000e-06,  -3.33800000e-06,  -3.33800000e-06,  -2.86100000e-06,
  -2.38400000e-06,  -2.38400000e-06,  -1.90700000e-06,  -1.90700000e-06,
  -1.43100000e-06,  -1.43100000e-06,  -9.54000000e-07,  -9.54000000e-07,
  -9.54000000e-07,  -9.54000000e-07,  -4.77000000e-07,  -4.77000000e-07,
  -4.77000000e-07,  -4.77000000e-07,  -4.77000000e-07,  -4.77000000e-07,
   0.00000000e+00]) * df / 2.0

    n = np.arange(- 0.5 * (N - 1), 0.5 * (N - 1) + 1)
    A = np.array([ h0 * 2.0 * np.cos(np.pi * (k + 0.5) * df / M * n * dt +       \
                               0.5 * np.pi * (k + 0.5) * (N - 1 - M ) / M) \
                for k in range(M)])
    S = np.array(A[:, ::-1], copy=True)
    A = A[:,:-1]
    S = S[:,:-1]
    N = N - 1


class Analyzer(object):
    """
    Analysis Filter Bank

    Compute the output of an array of causal FIR filters, critically sampled.

    Attributes
    ----------
    
    - `M`: number of subbands,

    - `N`: common filter length.

    Example
    -------

    We define an analysis filter bank with filters numbered from 0 to 3. 
    The i-th filter is a simple delay of i + 4 samples. 
    The common filter length is set to 8 (the minimal requirement).
    
        >>> import numpy as np
        >>> Z = np.zeros((4, 4), dtype=float)
        >>> I = np.eye(4, dtype=float)
        >>> a = np.c_[Z, I]

        >>> analyzer = Analyzer(a)
        >>> analyzer.M, analyzer.N
        (4, 8)
        >>> analyzer([1, 2, 3, 4])
        array([ 0.,  0.,  0.,  0.])
        >>> analyzer([5, 6, 7, 8])
        array([ 4.,  3.,  2.,  1.])
        >>> analyzer([0, 0, 0, 0])
        array([ 8.,  7.,  6.,  5.])
        >>> analyzer([0, 0, 0, 0])
        array([ 0.,  0.,  0.,  0.])
    """
    def __init__(self, a, dt=1.0, gain=1.0):
        """
        Arguments:
        ----------

          - `a`: filter bank impulse responses -- a two-dimensional numpy array 
            whose row `a[i,:]` is the impulse response of the `i`-th bank 
            filter.

          - `dt`: sampling time, defaults to `1.0`,

          - `gain`: a factor applied to the output values, defaults to `1.0`.
        """
        self.M, self.N = np.shape(a)
        self.A = gain * a * dt
        self.buffer = np.zeros(self.N)

    def __call__(self, frame):
        """
        Argument
        --------

        - `frame`: a sequence of `self.M` new input value of the filter bank, 
        
        Returns
        -------

        - `subbands`: the corresponding `self.M` new output subband values.
        """
        frame = np.array(frame, copy=False)
        if np.shape(frame) != (self.M,):
            raise ValueError("shape(frame) is not ({0},)".format(self.M))
        self.buffer[self.M:] = self.buffer[:-self.M]
        self.buffer[:self.M] = frame[::-1]
        return np.dot(self.A, self.buffer)


class Synthesizer(object):
    """
    Synthesis Filter Bank

    Combine critically sampled subband signals with an array of causal 
    FIR filters.

    Attributes
    ----------
    
    - `M`: number of subbands,

    - `N`: common filter length.

    Example
    -------

    We define a synthesis filter bank with filters numbered from 0 to 3. 
    The i-th filter is a simple delay of 7 - i samples. This synthesis
    filter bank provides a perfect reconstruction for the analysis filter
    bank implemented in the section "example" of the `Analyzer` 
    documentation, with a combined delay of 2 frames (8 samples).


        >>> Z = np.zeros((4, 4), dtype=float)
        >>> J = np.eye(4, dtype=float)[:,::-1]
        >>> a = np.c_[Z, J]

        >>> synthesizer = Synthesizer(a)
        >>> synthesizer.M, synthesizer.N
        (4, 8)
        >>> synthesizer([0, 0, 0, 0])
        array([ 0.,  0.,  0.,  0.])
        >>> synthesizer([4, 3, 2, 1])
        array([ 0.,  0.,  0.,  0.])
        >>> synthesizer([8, 7, 6, 5])
        array([ 1.,  2.,  3.,  4.])
        >>> synthesizer([0, 0, 0, 0])
        array([ 5.,  6.,  7.,  8.])
        >>> synthesizer([0, 0, 0, 0])
        array([ 0.,  0.,  0.,  0.])
    """
    def __init__(self, s, dt=1.0, gain=1.0):
        """
        Arguments:
        ----------

          - `s`: filter bank impulse responses -- a two-dimensional numpy array 
            whose row `s[i,:]` is the impulse response of the `i`-th bank 
            filter.

          - `dt`: sampling time, defaults to 1.0,

          - `gain`: a factor applied to the output values, defaults to `1.0`.
        """
        self.M, self.N = np.shape(s)
        self.P = np.transpose(gain * dt * s)[::-1,:]
        self.buffer = np.zeros(self.N)

    def __call__(self, frame):
        """
        Argument
        --------

        - `subbands`: a sequence of `self.M` new subband values, 
        
        Returns
        -------

        - `frame`: the corresponding `self.M` new output values.

        """
        frame = np.array(frame, copy=False)
        if np.shape(frame) != (self.M,):
            raise ValueError("shape(frame) is not ({0},)".format(self.M))
        self.buffer += np.dot(self.P, frame)
        output = self.buffer[-self.M:][::-1].copy()
        self.buffer[self.M:] = self.buffer[:-self.M]
        self.buffer[:self.M] = np.zeros(self.M)
        return output


def D(i, A=MPEG.A, S=MPEG.S, dt=MPEG.dt):
    """
    Distorsion Functions

    Arguments
    ----------
      - `i`: distorsion index (integer),
      - `A`: analysis filter bank, optional, defaults to `MPEG.A`,
      - `S`: synthesis filter bank, optional, defaults to `MPEG.A`,
      - `dt`: sample time, optional, defaults to `MPEG.dt`.

    Returns
    -------
      - `Di`: i-th distorsion function.
    """
    if np.shape(A)[0] != np.shape(S)[0]:
        error = "analysis and synthesis banks have a different number of bands"
        raise ValueError(error)
    M = np.shape(A)[0]
    if dt is None:
        dt = 1.0
    df = 1.0 / dt
    if not 0 <= i < M:
        raise ValueError("invalid distorsion index {0}".format(i))

    def Di(f, i=i, A=A, S=S, M=M, df=df, dt=dt):
        f = np.array(f, copy=False)
        Di_ = 0
        for k in range(M):
            Di_ = Di_ + F(A[k], dt=dt)(f) * F(S[k], dt=dt)(f + i * df/M)
        if i == 0:
            # T: delay induced by analysis + synthesis
            T = (0.5 * np.shape(A)[1] + 0.5 * np.shape(S)[1]) * dt 
            Di_ = Di_ - exp(-1j*2*pi*T*f)
        return Di_
    return Di

#
# Unit Tests
# ------------------------------------------------------------------------------
#
def test_FIR():
    """
Prerequisite: approximate equality

    >>> def match(x, y, EPS=1e-15):
    ...     return all(abs(x - y) < EPS)

FIR of order 0 (stateless): x2

    >>> fir = FIR([2.0])
    >>> all(fir.state == [])
    True
    >>> fir(3.0) == 6.0
    True
    >>> fir(-1.0) == -2.0
    True
    >>> all(fir([2.0, -0.5, 1.0]) == [4.0, -1.0, 2.0])
    True
    
FIR of order 2: linear extrapolation

    >>> fir = FIR([0.0, 2.0, -1.0])
    >>> all(fir.state == [0.0, 0.0])
    True
    >>> all(fir([1.0, 1.0, 1.0]) == [0.0, 2.0, 1.0])
    True
    >>> fir.state = [0.0, 0.0, 0.0] ##doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...
    >>> fir.state = [-1.0, 0.0] # most recent value first
    >>> fir(0.0)
    -2.0
    >>> all(fir.state == [0.0, -1.0])
    True
    >>> all(fir([0.0, 0.0]) == [1.0, 0.0])
    True
    >>> all(fir.state == [0.0, 0.0])
    True

FIR frequency response

    >>> fir = FIR([2.0]) # x 2.0
    >>> Fh = F(fir)
    >>> match(Fh([0.0, 0.25, 0.5]) , [2.0, 2.0, 2.0])
    True
    >>> fir = FIR([0.0, 1.0]) # delay
    >>> Fh = F(fir)
    >>> match(Fh([0.0, 0.25, 0.5]), [1.0, -1.0j, -1.0])
    True
    >>> fir = FIR([0.0, 2.0, -1.0]) # linear extrapolation
    >>> Fh = F(fir)
    >>> match(Fh([0.0, 0.5]), [1.0, -3.0])
    True
    >>> df = 8000.0; dt = 1.0 / df
    >>> fir = FIR([0.0, 2.0, -1.0]) # linear extrapolation
    >>> Fh = F(fir, dt=dt)

FIR linear properties w.r.t. (state, input)

    >>> fir = FIR([0.0, 1.0, -1.0]) # strictly causal delta
    >>> state = [0.5, -1.5]
    >>> fir.state = state
    >>> input = [-1.0, 1.0, -2.0]
    >>> output_1 = fir(input)
    >>> match(output_1, [2.0, -1.5, 2.0])
    True
    >>> fir.state = [0.0, 0.0]
    >>> output_2 = fir(input)
    >>> fir.state = state
    >>> output_2 += fir([0.0, 0.0, 0.0])
    >>> match(output_1, output_2)
    True
    """

def test_AR():
    """
Prerequisite: approximate equality

    >>> def match(x, y, EPS=1e-15):
    ...     return all(abs(x - y) < EPS)

AR of order 1: `y[n] = 2.0 * y[n-1] + u[n]`

    >>> ar = AR([2.0])
    >>> all(ar.state == [0.0])
    True
    >>> ar(3.0) == 3.0
    True
    >>> ar(-1.0) == 5.0
    True
    >>> all(ar([2.0, -0.5, 1.0]) == [12.0, 23.5, 48.0])
    True

AR of order 2: `y[n] = 0.75 * y[n-1] + 0.25 * y[n-2] + u[n]`

    >>> ar = AR([0.75, 0.25])
    >>> all(ar.state == [0.0, 0.0])
    True
    >>> ar(1.0) == 1.0
    True
    >>> all(ar.state == [1.0, 0.0])
    True
    >>> all(ar([0.25, 1.0]) == [1.0, 2.0])
    True
    >>> all(ar.state == [2.0, 1.0])
    True
    >>> ar.state = [0.0, 0.0, 0.0] ##doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...
    >>> ar.state = [-1.0, 0.0]
    >>> all(ar([0.0, 0.0]) == [-0.75, -0.8125])
    True

    # TODO: spectrum

    # TODO: AR linear properties wrt state, input
    """
    


