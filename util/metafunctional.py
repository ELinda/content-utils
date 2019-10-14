from functools import reduce
import numpy as np

def func_sum(*funcs):
    """
    return a unary function equivalent to the sum of arbitrary number of input unary functions
    example:
    func_sum(lambda x: x + 1, lambda x: x**2)(2) = 2 + 1 + 2**2 = 7
    """
    return lambda x: sum(f(x) for f in funcs)


# return a unary function equivalent to the product of arbitrary number of input unary functions
# e.g. func_sum(lambda x: x + 1, lambda x: x**2)(2) = (2 + 1) * 2**2 = 12
def func_prod(*funcs):
    base = lambda x: 1
    return reduce(lambda f1, f2: lambda x: f1(x) * f2(x), funcs, base)


# similar to func_prod, but scale func by factor, a constant value
def func_scale(func, factor):
    return lambda x: func(x * factor)


# return a unary function equivalent to the composition of arbitrary number of input unary functions
# Note that the rightmost function is the innermost in the function composition
# e.g. func_comp(lambda x: x + 1, lambda x: x**2)(2) = (2**2) + 1 = 5
def func_comp(*functions):
    identity = lambda x: x
    return reduce(lambda f1, f2: lambda x: f1(f2(x)), functions, identity)


# return a function that gives y
def func_y(y):
    return lambda x: y


def func_y_tile(y):
    from util.np_array import get_tiled
    return lambda x: get_tiled(y, len(x))


def get_smooth_env(fn, f0=lambda x: (x)**2, margin_factor=0.05):
    """
    input a function
    get back a function that is more smooth at the edges (no abrupt entr)
    """

    # transition points
    def get_env(x):
        c1, c2 = margin_factor, (np.max(x) - margin_factor)

        # factor to normalize on/offboarding
        factor = margin_factor / f0(margin_factor)
        f2 = lambda x: np.flip(f0((x-c2)*factor), axis=0)
        env = np.piecewise(x, [(x >= 0) &  (x < c1), (x >= c1) & (x < c2), x >= c2],
                           [lambda x: f0(x * factor), lambda x: 1.0, lambda x: f2(x)])
        return env
    return func_prod(get_env, fn)
