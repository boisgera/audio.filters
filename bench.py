#!/usr/bin/env python
import audio.filters

import docbench

def bench():
    """
    >>> import numpy as np
    >>> fir = audio.filters.FIR(a=np.ones(512))
    >>> input = np.ones(44100)
    >>> output = fir(input)
    """

if __name__ == "__main__":
    # Warning: the gc disabling can generate memory errors when a lot
    # of data is moved around. The gc should probably be not be disabled
    # by default.
    docbench.benchmod()
