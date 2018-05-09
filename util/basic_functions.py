from util.metafunctional import *
import numpy as np
######## sample functions, freq in Hz ########
# exponential decay function
def exp_decay(rate=4.0):
    return lambda x: np.exp(-x * rate)
def chirp(freq=1024.0):
    return lambda x: np.sin(2 * np.pi * freq * x ** 2)

def saw(freq=260.0):
    return lambda x: x * freq - np.floor(x * freq) - 0.5

def ring(freq=260.0):
    return lambda x: np.sin(2 * np.pi * freq * x)

def decaying_pluck_lower(freq=30.0):
    y1 = func_prod(exp_decay(rate=1.0), saw(freq=freq))
    y2 = func_prod(exp_decay(rate=1.0), ring(freq=freq))
    return func_sum(y1, y2)


def decaying_pluck(freq=60.0):
    y1 = func_prod(exp_decay(rate=6.0), saw(freq=freq))
    y2 = func_prod(exp_decay(), ring(freq=freq))
    return func_sum(y1, y2)


def normalized_max(data):
    data /= np.max(np.abs(data)) or 1
    return data


def seq_overlapping(fn=ring, freqs=[100, 200, 400, 200, 100], duration=1.0):
    N = len(freqs)
    block_size = N * duration
    return func_sum(*[func_prod(lambda t: 1 / N,
                                fn(fr),
                                lambda t, i=i: ((t > i / block_size) &
                                                (t < (i + 3) / block_size)).astype(int))
                      for i, fr in enumerate(freqs)])


def triangle(freq=60.0):
    """
    basically the absolute value of the
    """
    return func_comp(lambda x: normalized_max(x),
                     lambda x: np.abs(x) - .25, saw(freq=freq))


def array_fn(fn, args=[], kwargs={}):
    """return a lambda to apply a function onto an array"""
    return lambda x: fn(x, *args, **kwargs)


