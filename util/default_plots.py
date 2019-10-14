import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from util.constants import DEFAULT_SAMPLE_RATE
from util.np_array import split_into_subarrays_of_max_len


def plot(x, y):
    plt.plot(x, y, ls='none', marker='.')
    plt.show()


def first_index(arr, thres):
    # return first index passing a thres
    return np.argmax(arr >= thres) if thres and thres < arr[-1] else len(arr)


def plot_colormesh_corner(t, y, values, max_t=10, max_y=1500):
    """
    plot a heatmap where the colors are designated by "values"
    "duration" is the nunber of seconds represented on the x axis
    """
    max_y_index = first_index(y, max_y)
    max_t_index = first_index(t, max_t)
    plt.figure(figsize=(20, 4))
    plt.pcolormesh(t[:max_t_index], y[:max_y_index], values[:max_y_index, :max_t_index])
    plt.ylabel('y')
    plt.xlabel('t')
    plt.colorbar()
    plt.show()


def plot_and_play(y, sr=DEFAULT_SAMPLE_RATE, play=True, plot=True, shape=None):
    """
    y, any list-like object
    """
    if plot:
        num_points = np.size(y)
        num_secs = num_points / sr
        x = np.linspace(0, num_secs, num_points)
        plt.figure(figsize=shape or (max(num_secs, 20), 4))
        plt.plot(x, y)
        plt.show()
    if play:
        import sounddevice as sd
        sd.play(y, sr)


def spectrum(sig, t_incr=None, max_f_buckets=None):
    from scipy.fftpack import rfftfreq, rfft
    if t_incr is None:
        t_incr = 1.0 / DEFAULT_SAMPLE_RATE
    f_buckets = max_f_buckets or int(sig.size)
    print('computing spectrum for %s buckets for %s points' % (f_buckets, int(sig.size)))
    f = rfftfreq(f_buckets, d=t_incr)
    y = rfft(sig, f_buckets)
    y_norm = np.abs(y) / f_buckets * 2

    return f, y_norm


def plot_spectrum_by_segment(sig, t_incr=None, max_f_buckets=None,
                             max_f=None, max_len=DEFAULT_SAMPLE_RATE):
    """
    break into chunks of max_len and compute and plot spectrum for each
    """
    sig_segs = split_into_subarrays_of_max_len(sig, max_len)
    for sig_seg in sig_segs:
        f, y_norm = spectrum(sig_seg, t_incr=None, max_f_buckets=None)
        max_f_index = np.argmax(f > max_f) if max_f else len(f)
        plt.plot(f[:max_f_index], y_norm[:max_f_index])
        plt.grid(True)
        plt.show()


def plot_spec(y, sample_rate=DEFAULT_SAMPLE_RATE, max_freq=None,
              suppress_plot=False, void=False, take_log=True):
    """
    plot spectrogram (frequency on x axis and log of frequency presence on y axis)
    e.g. plot_spec(data, max_freq=2000)
    """
    f, t, Sxx = signal.spectrogram(y, sample_rate)
    if take_log:
        Sxx = np.log(Sxx + 0.0001)   # add small number to avoid divide-by-0
    if not suppress_plot:
        plot_colormesh_corner(t, f, Sxx, max_t=None, max_y=max_freq)
    if not void:
        return f, t, Sxx


def plot_dft(y, max_freq=None):
    A_rfft = np.fft.rfft(y)
    max_f_index = int(np.size(y) / 2) + 1
    f_rfft = range(max_f_index)
    max_freq = np.size(f_rfft) if max_freq is None else max_freq
    f = f_rfft[0:max_freq]
    A = np.abs(A_rfft[0:max_freq])
    plt.ylabel('A')
    plt.xlabel('f')
    plot(f, A)
    return f, A
