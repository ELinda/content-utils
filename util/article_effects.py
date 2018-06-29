import numpy as np
import librosa
import signal
import argparse

from util.io import write as write_to_same_directory_as_input
from util.constants import DEFAULT_SAMPLE_RATE
from util import librosa_based as lb
from util import np_array as na


def sig_handler(signum, frame):
    print('Seg Fault %s' % signum)


signal.signal(signal.SIGSEGV, sig_handler)


def get_non_quiet_segments(data, window_secs=0.25, slide_secs=0.125, threshold=0.00349,
                           sample_rate=DEFAULT_SAMPLE_RATE, debug=False):
    """ window = seconds in lookback window
        slide = how much the window slides forward each check
        threshold = the maximum average absolute value of data considered quiet
    """
    # number of times per increment to look back and check
    if debug:
        from util.default_plots import plot
    slide = int(slide_secs * sample_rate)
    window = int(window_secs * sample_rate)

    # indices at which to check backwards for average absolute value
    data_sample_pts = np.arange(window, np.size(data), slide)
    avgs = np.array([np.sum(np.abs(data[start - window:start])) / window
                     for start in data_sample_pts])

    if debug:
        print('avgs = %s\ndata_sample_pts= %s\nend=%s' % (
              avgs, data_sample_pts, np.size(data)))
        plot(data_sample_pts / sample_rate, avgs)

    return [data[data_sample_pts[i1] - window:data_sample_pts[i2]]
            for i1, i2 in na.merge_adjacent(np.where(avgs > threshold)[0])]


def get_intervals_from_dividers(divide_pts, last_sec):
    """ get an list of adjacent time intervals separated at the
    points specified by divide_pts
    e.g. [1, 2], 6 -> [(0, 1), (1, 2), (2, 6)]"""
    start_secs = np.concatenate([np.array([0]), divide_pts], 0)
    end_secs = np.concatenate([divide_pts, np.array([last_sec])], 0)

    return list(zip(start_secs, end_secs))


def get_stuttered_based_on_onsets(data, onsets, every=2,
                                  repeat=lambda: int(np.random.normal() * 2),
                                  sr=DEFAULT_SAMPLE_RATE):
    """ return stuttered version of the data based on onsets
    onsets are treated as the beginnings of some events (change in freq/amp, user defined)
    insert "repeat" repetitions of each data[index] every "every" indexes
    """
    last_sec = np.size(data) / sr
    between_onsets = get_intervals_from_dividers(onsets, last_sec)
    for i in np.flip(np.arange(0, len(between_onsets), every), 0):
        for _ in range(repeat()):
            between_onsets = np.insert(between_onsets, i, between_onsets[i], axis=0)

    segments = [data[int(start_sec * sr):int(stop_sec * sr)]
                for start_sec, stop_sec in between_onsets]
    return np.concatenate(segments)


def load_file_get_stuttered(full_path_and_file, stutter_fn=get_stuttered_based_on_onsets,
                            stutter_fn_args={}, write=True):
    """
    load the file specified. Process it, add stutter, concatentate, write out to same directory
    with slightly different name.
    """
    data_original, sr_original = lb.load_get_y_sr(full_path_and_file)
    print('\nloaded %s points with std dev %s' % (np.size(data_original), np.std(data_original)))

    # this is probably unnecessary but makes it easier (prevent passing extra arg)
    data = lb.resample_to_default(data_original[0:int(sr_original * 169)], sr_original)
    sample_rate = DEFAULT_SAMPLE_RATE
    print('\nresampled %s points from %s to %s' % (len(data), sr_original, sample_rate))

    non_quiet_segments = get_non_quiet_segments(data)
    segments_with_stutter = []
    for i, segment in enumerate(non_quiet_segments):
        segment_onsets = librosa.onset.onset_detect(segment, sample_rate, units='time')
        segment_replaced = stutter_fn(segment, segment_onsets, **stutter_fn_args)
        segments_with_stutter.append(segment_replaced)
        if i % 40 == 0:
            print('\nprocessed %s of %s segments' % (i + 1, len(non_quiet_segments)))

    result = np.concatenate(segments_with_stutter)

    if write:
        write_to_same_directory_as_input(full_path_and_file, result, 'stuttered')
    return result


if __name__ == "__main__":
    description = 'produce woven version of input files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='input files')

    for file in parser.parse_args().files:
        print('processing %s' % file)
        load_file_get_stuttered(file)
