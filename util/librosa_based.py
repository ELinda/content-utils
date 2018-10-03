import librosa
from util.constants import DEFAULT_SAMPLE_RATE


def load_get_y_sr(file):
    """ return the data and sample rate of a file
    probably ogg, mp3, wav would work. uses pysoundfile for ogg"""
    from numpy import ndarray
    import soundfile as sf
    if file.split('.')[1] == 'ogg':
        y, sr = sf.read(file)
        # take the first channel if there are multiple
        if len(y) > 0 and type(y[0]) == ndarray:
            y = y[:, 0]
    else:
        y, sr = librosa.load(file)
    return y, sr


def load_data_with_sr(file, target_sr=DEFAULT_SAMPLE_RATE,
                      keep_secs=None):
    """ same as above but possible to resample if target and initial
        sr are different
        file: location as a string e.g. /path/to/file.ogg """
    y, sr = load_get_y_sr(file)
    if keep_secs:
        y = y[:sr * keep_secs]
    if sr != target_sr:
        print('resample %s pts from %s to %s rate' % (len(y), sr, target_sr))
        y = librosa.core.resample(y, sr, target_sr)
    return y


def resample_to_default(data, orig_sr):
    return librosa.core.resample(data, orig_sr, DEFAULT_SAMPLE_RATE)


def shift(x, steps, sample_rate=DEFAULT_SAMPLE_RATE, steps_per_octave=12):
    return librosa.effects.pitch_shift(x, sr=sample_rate,
                                       n_steps=steps,
                                       bins_per_octave=steps_per_octave)


def stretch(x, factor):
    return librosa.effects.time_stretch(x, factor)

