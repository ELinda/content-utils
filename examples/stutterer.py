import argparse
from random import randint

import soundfile as sf
import numpy as np
from scipy import signal


def get_avg_freqs(f, Sxx):
    """get average frequency per time bin, weighted by magnitude found in Sxx[f, t]"""
    return [sum([f_i * m for f_i, m in zip(f, Sxx[:,t_i])])/sum(Sxx[:,t_i])
            for t_i in range(Sxx.shape[1])]

def rep_start_end(data_segment, start_rep=4, end_rep=4, samples=1984):
    """repeat start and end, of width equal to `samples` """
    return np.concatenate(  [data_segment[0:samples]] * start_rep
                          + [data_segment]
                          + [data_segment[-samples:]] * end_rep)

def get_peak_onsets(keys, values, thres):
    """ get keys where values are greater than thres,
    omitting keys occurring contiguously after an onset key"""
    last_peak = False
    result = []

    for k, v in zip(keys, values):
        if v > thres and not last_peak:
            result.append(k)

        last_peak = v > thres

    return result

def secs2i(seconds, sr):
    return int(seconds * sr)

def get_stuttered(data, sr, thres=2000):
    f, t, Sxx = signal.spectrogram(data, sr)

    avg_freqs = get_avg_freqs(f, Sxx)
    
    f_peak_onsets = get_peak_onsets(t, avg_freqs, thres)

    # convert second time unit into sample index
    f_peak_onsets_i = [secs2i(t, sr) for t in f_peak_onsets]

    # intervals whose ends will repeat
    intervals = list(zip([0] + f_peak_onsets_i, f_peak_onsets_i + [len(data)]))

    stuttered = np.concatenate([rep_start_end(data[a:b], randint(1,2), randint(1,2))
                                for a, b in intervals])
    return stuttered


if __name__ == "__main__":
    # get the argument values passed, and process them
    parser = argparse.ArgumentParser(description='stuttered')
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='input files')
    files = parser.parse_args().files

    for file in files:
        data, sr = sf.read(file)
        if len(data.shape) == 2:
            stuttered = list(zip(*(get_stuttered(data[:, i], sr) for i in [0, 1])))
        else:
            stuttered = get_stuttered(data, sr)
        file_parts = file.split('.')
        new_file_name = '%s%s%s' % ('.'.join(file_parts[:-1]), '_stut.', file_parts[-1])
        print('write %s' % (new_file_name))
        sf.write(new_file_name, stuttered, sr)


