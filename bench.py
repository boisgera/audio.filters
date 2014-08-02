#!/usr/bin/env python
import audio.filters

import docbench

def bench():
    """
    >>> import numpy as np
    >>> fir = audio.filters.FIR(0.1 * np.ones(100))
    >>> input = np.ones(100000)
    >>> output = fir(input)
    """

if __name__ == "__main__":
    docbench.benchmod()
