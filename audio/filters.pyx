
# Python 2.7 Standard Library

# Third-Party Libraries
import numpy as np
cimport numpy as np
cimport cython

# TODO:
#
#  - rethink the base class issue. Interface with the common concepts
#    (parameters, state, etc) but also specialized implementations that
#    are fast. Focus on specialized and fast implementation first and foremost.
#    opaque state: 0/None state (?), save, restore, that's it ?
#
#  - "fixed-width" filters.Simpler is better, the previous design was too smart.
#    No check is done that we don't screw things up, we may give access to 
#    internal state (for performance reasons).
#
#  - concept of opaque state that can be obtained, restored, maybe with context
#    manager that can be used to deal for a moment with a state and then 
#    restore the previous one (inc. if some exception happens).
#   
#  - types of filters: FIR, AR, ARMA, but also lattice filters (for voice apps).

cdef class Filter:
     pass

cdef class FIR(Filter):

    cdef public object a
    cdef public object state
    cdef double[:] _a
    cdef double[:] _state

    def __cinit__(self, a):
        self.a = np.array(a)
        self._a = self.a
        self.state = np.zeros(len(a)-1)
        self._state = self.state

    @cython.boundscheck(False)
    def __call__(self, input):
        cdef unsigned int i, j, m, n
        cdef double[:] _input, _output
        cdef double[:] _a, _state

        if np.isscalar(input):
            output = self.a[0] * input + np.dot(self.a[1:], self.state)
            self.state = np.r_[input, self.state[:-1]]
            return output
        else:
            _input = np.array(input, dtype=float, copy=False)
            _a = self._a
            _state = self._state
            m = _input.shape[0]
            n = _state.shape[0]
            output = np.empty(m)
            _output = output
            for i in range(m):
                _output[i] = _a[0] * _input[i] 
                for j in range(n):
                    _output[i] += _a[j+1] * _state[j]
                _state[1:] = _state[:-1]
                _state[0] = _input[i]
        return output



