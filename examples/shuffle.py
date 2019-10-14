import os
import argparse

import random
import numpy as np

import soundfile as sf
from util.article_effects import get_intervals_from_dividers
import librosa

from util.replacement_effects import secs2i
from util.article_effects import get_intervals_from_dividers
from util.io import write


def write2(full_path_and_file, data, sr, new_suffix=''):
    """ write a file to the path specified. format is inferred from original"""
    directory = os.path.dirname(full_path_and_file)
    file_parts = full_path_and_file.split('.')
    new_file_name = ''.join(file_parts[:-1]) + '_' + new_suffix + '.' + file_parts[-1]
    new_file_path = os.path.join(directory, new_file_name)
    print('Output %s points to %s' % (len(data), new_file_path))
    sf.write(new_file_path, data, sr)


def output_h_p(file):
    data, sr = librosa.load(file)
    data_stft = librosa.stft(data)
    data_stft_harmonic, data_stft_percussive = librosa.decompose.hpss(data_stft)
    data_harm = librosa.istft(data_stft_harmonic)
    data_perc = librosa.istft(data_stft_percussive)
    write(file, data_harm, 'h', None, sr)
    write(file, data_perc, 'p', None, sr)

def output_shuf(file):
    data, sr = librosa.load(file)
    onsets_bt = librosa.onset.onset_detect(data, sr, units='time', backtrack=True)
    segs = [data[secs2i(a):secs2i(b)] for a, b in get_intervals_from_dividers(onsets_bt, float(len(data))/sr)]
    shuffled = np.concatenate(sorted(segs, key=lambda x: random.random()))

    write(file, shuffled, 'shuf', None, sr)


if __name__ == "__main__":
    description = 'replacement'
    # get the argument values passed, and process them
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='input files')
    files = parser.parse_args().files

    for file in files:
        output_shuf(file)
