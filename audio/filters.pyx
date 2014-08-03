
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
#    manager (`tmp_state` ?) that can be used to deal for a moment with a 
#    state and then restore the previous one (inc. if some exception happens).
#   
#  - types of filters: FIR, AR, ARMA, but also lattice filters (for voice apps).

cdef class Filter:
     pass

cdef class FIR(Filter):
    cdef readonly object a # Arrays are not allowed here, hence the 'object' type.
    cdef double[:] _state  # Sequence of input values, with oldest values first.

    def __cinit__(self, a):
        self.a = np.array(a, dtype=float, copy=True)
        self._state = np.zeros(len(self.a)-1)

    cpdef np.ndarray[np.float64_t, ndim=1] get_state(self):
        cdef np.ndarray[np.float64_t, ndim=1] state
        state = np.empty(self._state.shape[0]) 
        state[:] = self._state
        return state

    #
    # State API: 
    # 
    #   - the state shall be managed as an opaque object, 
    #   - get / set operations do copy the state,
    #   - set_state(None) resets the filter state.
    # 
 

    cpdef set_state(self, double[:] state):
        if state is None:
            self._state = np.zeros(len(self.a)-1)            
        else:
            self._state = state.copy()

    property state:
        def __get__(self):
            return self.get_state()

        def __set__(self, value):
            self.set_state(value)

    def __call__(self, input):
        # TODO: support scalar inputs
        cdef np.ndarray[np.float64_t, ndim=1] _input
        _input = np.array(input, dtype=float, copy=False)
        return FIR_filter(self.a, self._state, _input)
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim=1] FIR_filter(double[:] a, double[:] state, double[:] input):
    cdef unsigned int i, j, m, n
    cdef np.ndarray[np.float64_t, ndim=1] output
    cdef double[:] _output
    cdef double tmp_output   
    cdef double[:] ext_state
        
    m = input.shape[0]
    n = state.shape[0]
    ext_state = np.empty(m+n)
    ext_state[:n] = state
    output = np.empty(m)
    _output = output
    for i in range(m):
        tmp_output = a[0] * input[i]
        for j in range(n):
            tmp_output += a[j+1] * ext_state[n-1-j+i]
        output[i] = tmp_output
        ext_state[n+i] = input[i]
    state[:] = ext_state[m:m+n]
    return output


