
# Python 2.7 Standard Library

# Third-Party Libraries
import numpy as np

# TODO:
#
#  - rethink the base class issue. Interface with the common concepts
#    (parameters, state, etc) but also specialized implementations that
#    are fast. Focus on specialized fast implementation first and foremost.
#
#  - "fixed-width" filters.Simpler is better, the previous design was too smart.
#    No check is done that we don't screw things up, we may give access to 
#    internal state for performance reasons
#
#  - concept of opaque state that can be obtained, restored, maybe with context
#    manager that can be used to deal for a moment with a state and then 
#    restore the previous one (inc. if some exception happens).
#   
#  - types of filters: FIR, AR, ARMA, but also lattice filters (for voice apps).

class Filter(object):
     pass

class FIR(Filter):
    def __init__(self, a):
        self.a = np.array(a)
        self.state = np.zeros(len(a)-1)
    def __call__(self, input):
        if np.isscalar(input):
            output = self.a[0] * input + np.dot(self.a[1:], self.state)
            self.state = np.r_[input, self.state[:-1]]
        else:
            output = np.array([self(_input) for _input in input])
        return input


