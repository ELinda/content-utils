import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from util.constants import DEFAULT_SAMPLE_RATE


def plot(x, y):
    plt.plot(x, y, ls='none', marker='.')
    plt.show()


def plot_and_play(y, play=True):
    """
    y, any list-like object
    """
    num_points = np.size(y)
    num_secs = num_points / DEFAULT_SAMPLE_RATE
    x = np.linspace(0, num_secs, num_points)
    plt.plot(x, y)
    plt.show()
    if play:
        import sounddevice as sd
        sd.play(y, DEFAULT_SAMPLE_RATE)


def spectrum(sig, t_incr=None, max_f_buckets=None):
    from scipy.fftpack import rfftfreq, rfft
    if t_incr is None:
        t_incr = 1.0 / DEFAULT_SAMPLE_RATE
    f_buckets = max_f_buckets or int(sig.size)
    print('computing spectrum for %s buckets for %s points' % (f_buckets, int(sig.size)))
    f = rfftfreq(f_buckets, d=t_incr)
    y = rfft(sig,f_buckets)
    y_norm = np.abs(y) / f_buckets * 2

    return f, y_norm

def get_spec(y, sample_rate=DEFAULT_SAMPLE_RATE):
    f, t, Sxx = signal.spectrogram(y, sample_rate)
    return f, t, Sxx


def plot_spec(y, sample_rate=DEFAULT_SAMPLE_RATE, max_freq=None):
    f, t, Sxx = signal.spectrogram(y, sample_rate)
    max_freq_index = np.argmax(f>max_freq) if max_freq else len(f)
    plt.pcolormesh(t, f[0:max_freq_index], Sxx[0:max_freq_index,:])
    plt.ylabel('f')
    plt.xlabel('t')
    plt.colorbar()
    plt.show()
    return f, t, Sxx


def get_f_y():
    f = rfftfreq(int(sig.size/2), d=t[1]-t[0])
    y = rfft(sig,int(sig.size/2))
    return f, y 


def plot_dft(y, max_freq=None):
    A_rfft = np.fft.rfft(y)
    max_f_index = int(np.size(y)/2) + 1
    f_rfft = range(max_f_index)
    max_freq = np.size(f_rfft) if max_freq is None else max_freq
    f = f_rfft[0:max_freq]
    A = np.abs(A_rfft[0:max_freq])
    plt.ylabel('A')
    plt.xlabel('f')
    plot(f, A)
    return f, A
