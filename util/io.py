import os
import datetime
import soundfile as sf
from util.constants import DEFAULT_SAMPLE_RATE


def write(full_path_and_file, data, new_suffix=None, form=None, sr=DEFAULT_SAMPLE_RATE):
    """ write a file to the path specified. we will try to infer the encoding by suffix,
    unless "form" is specified
    /old/path/old_file_name_new_suffix.form"""
    directory = os.path.dirname(full_path_and_file)
    file_parts = full_path_and_file.split('.')
    if new_suffix is None:
        new_suffix = datetime.datetime.today().strftime('%Y%m%d')
    file_prefix = ''.join(file_parts[:-1]) + '_' + new_suffix
    # keep same extension if no format was specified
    file_ext = file_parts[-1] if not form or form.lower() == file_parts[-1] else form.lower()
    new_file_name = os.path.join(directory, file_prefix + '.' + file_ext)
    print('Intent to output %s points to %s' % (len(data), new_file_name))
    if form:
        sf.write(new_file_name, data, sr, format=form)
    else:
        sf.write(new_file_name, data, sr)


def write_all(base_dir, prefix, data_list, extension='ogg'):
    """
    e.g. write_with_prefix(base_dir, prefix='quack', data_list=list_of_4_arrays)
    results in output to the following four files of content in data_list:
    base_dir/quack_0.ogg
    base_dir/quack_1.ogg
    base_dir/quack_2.ogg
    base_dir/quack_3.ogg
    """
    for i, data in enumerate(data_list):
        full_path_and_file = os.path.join(base_dir, '%s.%s' % (prefix, extension))
        write(full_path_and_file, data, new_suffix=str(i))
