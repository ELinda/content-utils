import os
import argparse
import itertools
import numpy as np

from util.librosa_based import load_data_with_sr
from util.article_effects import get_non_quiet_segments
from util.io import write as write_to_same_directory_as_input
from util.constants import DEFAULT_SAMPLE_RATE
from util.np_array import split_into_subarrays_of_max_len


def split_all_sublists_over_max_len(L, max_len=DEFAULT_SAMPLE_RATE, debug=False):
    """ return copy of input list L (a list of arrays), with replacement
    of any element that's an array of more than max_len length,
    with multiple arrays in the same position,
    so that the concatenation of the new arrays is the same as the original
    e.g. [[3, 13, 7], [2, 1, 32, 4, 3, 10]], 3
    ------> [[3, 13, 7], [2,  1, 32], [4,  3, 10]]
    """
    L_copy = L.copy()
    for i, arr in enumerate(L_copy):
        subarrays = split_into_subarrays_of_max_len(arr, max_len)
        if len(subarrays) > 1:
            L_copy[i:i + len(subarrays)] = subarrays
            if debug:
                print('replaced one array with %s at index %s' % (len(subarrays), i))
    return L_copy


def write_or_return_merged_data(files, write=True):
    """
    load data from all the files and weave them, trying to set breakpoints at silent eras
    assumes that files is a non-empty array of file names to process
    """
    article_names_datas = [(file_name, load_data_with_sr(file_name)) for file_name in files]
    # a list of arrays of maximum length max_len, so that their concat is the original
    # one list of lists per source
    significant_segs = [split_all_sublists_over_max_len(get_non_quiet_segments(data),
                                                        max_len=int(DEFAULT_SAMPLE_RATE / 2))
                        for name, data in article_names_datas]
    # itertools.zip_longest to get a list of tuples, each comprised of one segment per source
    # convert the tuples of arrays to lists of arrays, so that any bottom-level Nones
    # are converted to empty arrays. Concatenate each list of segments and then everything
    list_of_list_of_arr = [[arr if arr is not None else np.array([]) for arr in tup]
                           for tup in itertools.zip_longest(*significant_segs)]
    woven = np.concatenate([np.concatenate(list_of_arr) for list_of_arr in list_of_list_of_arr])

    if write:
        new_directory = os.path.dirname(files[0])
        file_prefix = '_'.join(['.'.join(file.split('.')[:-1]) for file in files])
        full_file_name = os.path.basename(file_prefix) + '.wav'  # other codecs not always working
        file_pattern = os.path.join(new_directory, full_file_name)
        print('output to time-stamped: %s' % (file_pattern))
        write_to_same_directory_as_input(os.path.join(new_directory, full_file_name), woven)
    else:
        return woven


if __name__ == "__main__":
    description = 'produce woven version of input files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='input files')
    files = parser.parse_args().files
    write_or_return_merged_data(files, write=True)
