import os
import math
import time
import numpy as np
import random
import argparse

import librosa

from util.librosa_based import load_data_with_sr
from util import basic_functions as bf
from util.default_plots import plot_and_play as pp
from util.librosa_based import stretch, shift
from util.replacement_effects import get_binned_freqs_mags, get_merged_intervals
from util.constants import DEFAULT_SAMPLE_RATE as dsr
from util.article_effects import write_to_same_directory_as_input
from util.basic_functions import decaying_pluck_lower as dpl
from util.filters import butter_band_filter


def pl(y):
    pp(y, plot=False)


def t_for_dur(seconds=1.0):
    """
    return time axis from 0 to "seconds"
    so that each second has "dsr" points
    """
    return np.linspace(0, seconds, seconds * dsr)


def repeated(data, times=3):
    return np.tile(data, times)


def modified_dpl(freq=120, seconds=0.75, stretch_factor=0.6):
    return stretch(overlay_decaying(dpl(freq=freq)(t_for_dur(seconds=seconds)), num_times=4),
                   stretch_factor)


def secs2i(seconds, sr=dsr):
    return int(seconds * sr)


def low_honk_seq(os_honk, honk_shift_amt=8, honk_overlay_factor=2,
                 prefix_ratio=.5, gap_sec=.7):
    # sequence of the same low honk, then the honk prefix repeated a few times
    low_h3 = os_honk(honk_shift_amt, honk_overlay_factor)
    low_h3_end = low_h3[:int(len(low_h3) * prefix_ratio)]
    return concat(low_h3, quiet(gap_sec), low_h3_end, quiet(gap_sec), low_h3_end, low_h3_end)


def high_honk_seq(os_honk, shift_amt=0, overlay_factor=0, cutoffs=(20, 2000)):
    s, o = shift_amt, overlay_factor
    honkseq = np.concatenate([os_honk(s, o, 0), os_honk(s, o, 1), os_honk(s, o, 0),
                              os_honk(s, o, 0), quiet(.3), os_honk(s, o, 0),
                              os_honk(s, o, 0)[:1000],
                              stretch(np.flip(os_honk(s + 1, o + 3, 0)[:1000], axis=0), .2)])
    return butter_band_filter(honkseq, cutoffs)


def high_honk_seq_2(os_honk, shift_amt=0, overlay_factor=0, cutoffs=(500, 2500)):
    s, o = shift_amt, overlay_factor
    stretched_high_honk = stretch(np.flip(os_honk(s + 1, o + 3, 0)[:1000], axis=0), .2)
    honkseq = np.concatenate([os_honk(s - 1, o, 0), os_honk(s - 2, o, 1), os_honk(s - 3, o, 0),
                              os_honk(s, o + 2, 0), quiet(.3), os_honk(s, o + 2, 0),
                              os_honk(s - 0.5, o + 2, 0)[:1000],
                              stretched_high_honk, quiet(.3),
                              stretched_high_honk,
                              shift(stretched_high_honk, 4),
                              shift(stretched_high_honk, 6),
                              quiet(.3), shift(stretched_high_honk, 6),
                              shift(stretched_high_honk, 4)])
    return butter_band_filter(honkseq, cutoffs)


def high_honk_seq_3(shift_amt=0, overlay_factor=0, cutoffs=(500, 2500)):
    s, o = shift_amt, overlay_factor
    stretched_high_honk = stretch(np.flip(os_honk(s + 1, o + 3, 0)[:1000], axis=0), .2)
    honkseq = np.concatenate([shift(stretched_high_honk, 4),
                              shift(stretched_high_honk, 6),
                              quiet(.3), shift(stretched_high_honk, 6),
                              shift(stretched_high_honk, 4)])
    return butter_band_filter(honkseq, cutoffs)


def get_sig_freq_time_list(data, threshold=0.15):
    fft_freqs, pitches, magnitudes = get_binned_freqs_mags(data / np.max(data), threshold=threshold)
    info = get_merged_intervals(pitches, magnitudes)
    t = np.linspace(0, len(data) / dsr, magnitudes.shape[1])
    info.sort(key=lambda x: x['start'])
    return info, t, fft_freqs


def get_high_mag_pitch_avg(data, percentile=90):
    """return average of the pitches of magnitude whose percentile exceeds "percentile"
    use it as a representative frequency
    """
    info_entries, t, fft_freqs = get_sig_freq_time_list(data)
    mag_of_percentile = np.percentile([info['mag'] for info in info_entries], percentile)
    above_percentile_pitches = [info['pitch'] * info['mag'] for info in info_entries
                                if info['mag'] >= mag_of_percentile]
    sum_of_mags_above_percentile = sum([info['mag'] for info in info_entries
                                        if info['mag'] >= mag_of_percentile])
    avg_of_those_pitches = np.sum(above_percentile_pitches) / sum_of_mags_above_percentile
    return avg_of_those_pitches


def get_shift_factor(source_freq, target_freq, bins_per_oct=24):
    # target_freq = source_freq * 2 ** (n_bins/bins_per_oct)
    # solve for n_bins (following is log base 2)
    # n_bins = np.log(target_freq / source_freq, 2) * bins_per_oct
    return math.log(target_freq / source_freq, 2) * bins_per_oct


def shift_to_target(data, source_freq, target_freq, bins_per_oct=24):
    shift_amount = get_shift_factor(source_freq, target_freq, bins_per_oct)
    return shift(data, shift_amount)


def exp_dec(data, onset_ratio=0.1, decay_ratio=0.7):
    x = np.linspace(0, len(data), len(data))
    onset_pt = onset_ratio * len(x)
    decay_pt = decay_ratio * len(x)
    piece_conditions = [x < onset_pt,
                        (x >= onset_pt) & (x < decay_pt),
                        x >= decay_pt]
    piece_forms = [lambda x: x / onset_pt,
                   lambda x: 1,
                   lambda x: np.exp(-10 * (x - decay_pt) / dsr)]
    dot_product_operand = np.piecewise(x, piece_conditions, piece_forms)
    return data * dot_product_operand


def get_note_of_length(data, seconds, break_ratio=0.8):
    break_pt = int(break_ratio * len(data))
    current_seconds = len(data) / dsr
    target_seconds_left = seconds - (current_seconds * break_ratio)
    current_seconds_left = (1.0 - break_ratio) * len(data) / dsr
    if current_seconds > seconds:
        data_until_desired = data[:secs2i(seconds)]
        return exp_dec(data_until_desired)

    # this can be greater or less than 1.0. less than 1.0 = longer
    stretch_factor = current_seconds_left / target_seconds_left
    data_ext = np.concatenate([data[:break_pt],
                               stretch(data[break_pt:], stretch_factor)])

    return exp_dec(data_ext)


def get_note_of_freq(data, new_freq, old_freq=None):
    """
    Resample data by factor so that the average of its louder freqs is "new_freq"
    Note: this is half taken from librosa shift function in effects.py """
    if not old_freq:
        old_freq = get_high_mag_pitch_avg(data)
    # e.g. 220 -> 440 Hz equates to a rate (stretching factor) of 2
    rate = old_freq / new_freq

    # Stretch in time, then resample
    data_shift = librosa.core.resample(stretch(data, rate), float(dsr) / rate, dsr)

    # Crop to the same dimension as the input
    return librosa.util.fix_length(data_shift, len(data))


def get_bday_note_info():
    """ get list of tuples: shift amount in semitones, and relative duration"""
    ver1 = [(-4, 1), (-4, 1), (-2, 1.5), (-4, 1), (1, 1), (0, 1.5)]
    ver2 = [(-4, 1), (-4, 1), (-2, 1.5), (-4, 1), (3, 1), (1, 1.5)]
    ver3 = [(-4, 1), (-4, 1), (8, 1.5), (5, 1), (1, 1), (0, 1.5), (-2, 1.5)]
    ver4 = [(10, 1), (10, 1), (9, 1.5), (5, 1), (7, 1), (5, 1.5)]
    vers = ver1 + ver2 + ver3 + ver4
    return vers


def get_bday_2(all_data):
    vers = get_bday_note_info()

    # get old freqs to memoize the freq info of each data
    old_freqs = [get_high_mag_pitch_avg(data) for data in all_data]
    base_freq = min(old_freqs)
    print('freq info')
    print(base_freq)
    print(old_freqs)

    def get_single_note(i_data, shift_steps, seconds):
        data = all_data[i_data]
        old_freq = old_freqs[i_data]
        new_freq = base_freq * 2 ** (shift_steps / 12)
        break_ratio = 0.5 if len(data) < dsr / 4 else 0.7
        return get_note_of_length(get_note_of_freq(data, new_freq, old_freq),
                                  seconds, break_ratio=break_ratio)

    notes = [np.concatenate([
             get_single_note(random.randrange(0, len(all_data)), ss, dt / 3),
             get_single_note(random.randrange(0, len(all_data)), ss, dt / 15) * .5
             ]) for ss, dt in vers]
    return np.concatenate(notes)


def get_bday(data_1, data_2):
    vers = get_bday_note_info()

    def get_single_note(data_1, data_2, shift_steps, seconds):
        return np.concatenate([shift(get_note_of_length(data_2, seconds), shift_steps * 2),
                               quiet(0.05)] +
                              ([shift(data_1[:int(random.random() * len(data_1))], shift_steps * 2)]
                               if random.random() > .5 else []))
    notes = [get_single_note(data_1, data_2, ss, dt) for ss, dt in vers]
    all_notes = np.concatenate([data_1] + notes)
    return all_notes


if __name__ == "__main__":
    description = 'plesae provide a honk file pattern to construct birthday verses'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pattern', type=file, action='store', nargs='+')
    #base_path = '/Users/cail029/Documents/utility/geese/'
    honks = []
    for file in parser.parse_args().files:
        honks.append(load_data_with_sr(file))
    #honks = [load_data_with_sr(os.path.join(base_path, 'honk_%s.ogg' % i)) for i in range(8)]
    honks = [honk / np.max(honk) * 0.8 for honk in honks]
    geese = load_data_with_sr(os.path.join(base_path, 'geese.mp3'))
    fft_freqs, pitches, magnitudes = get_binned_freqs_mags(geese, threshold=0.15)
    ode_shifts = [2, 0, -2, -4, -6, -8, -10, -12, -8, -6, -6, -4, -4, -2]
    ode = np.concatenate([stretch(shift(honks[0], x), .4) for x in ode_shifts])

    t = np.linspace(0, 0.5, 0.5 * dsr)
    abscos = np.abs(np.cos(60 * np.pi * t) + np.cos(70 * np.pi * t) + 2) / 4
    h2_mod = (shift(stretch(honks[2], 0.75), -12 * 7)[:16000])
    h1_mod = shift(stretch(honks[1], 0.25), -12 * 7)[:11000]

    dpl_rep = np.concatenate([repeated(modified_dpl(freq=120), 3), modified_dpl(freq=60)], axis=0)

    def quiet(duration=1.0):
        return bf.triangle(freq=80)(np.linspace(0, duration, duration * dsr)) * .01

    def os_honk(shift_amt, overlay_factor):
        return overlay_decaying(shift(honks[3], -6 * shift_amt),
                                num_times=1 + overlay_factor * 2)

    def concat(*args):
        return np.concatenate(args)

    seq = np.concatenate([h2_mod, quiet(.8),
                          h2_mod[:secs2i(.2)], quiet(.8), h2_mod])
    cutting = np.concatenate([butter_lowpass_filter(np.tile(seq, 4),
                                                    cutoffs=[50 * x, 500 + 100 * x])
                              for x in range(4)])

    h1 = os_honk(8, 2)
    h1rep = concat(h1, quiet(.7), h1[:7000], quiet(.7), h1[7000:], h1[7000:])
    # os_honk(2, 0), quiet
    bd = get_bday_2([h for i, h in enumerate(honks) if i in [4, 5, 6, 7]])
    write_to_same_directory_as_input('/Users/cail029/Documents/utility/bd.wav', bd)
    import pdb; pdb.set_trace()
