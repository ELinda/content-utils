import numpy as np


def flip(x):
    return np.flip(x, axis=0)


def set_segments_to_value(arr, segments, value=0):
    """ Modifies arr so that segments are set to value
    e.g. [(None, None)], 0 --> array becomes all 0"""
    for segment in segments:
        arr[segment[0]:segment[1]] = value


def overlay_decaying(data, num_times=5, decay_ratio=.8, delay_gap=500):
    max_size = data.size
    print(max_size)

    def get_padded(before, middle, after):
        return np.concatenate([np.zeros(before), middle, np.zeros(after)])

    return np.sum([.8 ** x * get_padded(delay_gap * x, data, delay_gap * (num_times - x))
                   for x in range(num_times)], axis=0)


def fft_fun(data, func=set_segments_to_value, fn_args=[[(1400, None)]]):
    """
    perform function "func" on the FFT,
    assuming this function takes the fft results of the data
    and the rest of the arguments are in fn_args
    return the inverse (so e.g. if the identity function is given,
    you will get the original data back, since nothing is done on fft results)
    """
    from numpy.fft import rfft, irfft

    rfft_data = rfft(data)
    func(rfft_data, *fn_args)
    return irfft(rfft_data)


def normalized_max(data):
    data /= np.max(np.abs(data)) or 1
    return data


def merge_adjacent_rec(a, start=None, end=None):
    """
    this can cause a max recursion depth error!
    """
    if len(a) == 0:
        return [(start, end)]
    elif end and a[0] == end + 1:
        return merge_adjacent_rec(a[1:], start, a[0])
    elif start and end:
        new_ends = a[0] if len(a) else None
        return [(start, end)] + merge_adjacent_rec(a[1:], new_ends, new_ends)
    else:
        return merge_adjacent_rec(a[1:], a[0], a[0])


def merge_adjacent(arr):
    """ merge any numbers that are adjacent to one another
    creating intervals representing contiguous values
    # e.g. merge_adjacent([10,11,12,19,20,21,35])
    #                  -> [(10, 12), (19, 21), (35, 35)]
    """
    start = arr[0]
    end = arr[0]
    i = 1
    merged = []

    while i < len(arr):
        if arr[i] == end + 1:
            end = arr[i]
        else:
            merged.append((start, end))
            start = arr[i]
            end = arr[i]
        i = i + 1

    merged.append((start, end))
    return merged


def get_tiled(data, to_fit_size):
    """
    get a tiling of the data
    1. repeat "data" enough times to have at least "to_fit_size" records
    2. then, truncate it to "to_fit_size" and return this
    """
    repeat_times = int(np.ceil(to_fit_size / len(data)))
    repeated = np.tile(data, repeat_times)
    return repeated[0:to_fit_size]


def replaced(L, old, new):
    """ return a copy of L, same, but all "old" in replaced with "new" """
    return [x if x != old else new for x in L]


def prefix_all(value, LL):
    """ for LL a list of lists, prefix all with value """
    return [[value] + L for L in LL]


def all_permutations(lst):
    """
    this is copied from stack overflow and it returns
    all the permutations of the given list "lst"
    it worked better than the one I came up with before
    """
    # If lst is empty then there are no permutations
    if len(lst) == 0:
        return []
    # If there is only one element in lst then, only
    # one permuatation is possible
    if len(lst) == 1:
        return [lst]
    # Find the permutations for lst if there are
    # more than 1 characters

    result = []  # empty list that will store current permutation
    # Iterate the input(lst) and calculate the permutation
    for i in range(len(lst)):
        m = lst[i]
        # Extract lst[i] or m from the list.  remLst is remaining list
        remLst = lst[:i] + lst[i + 1:]

        # Generating all permutations where m is first element
        for p in all_permutations(remLst):
            result.append([m] + p)
    return result
