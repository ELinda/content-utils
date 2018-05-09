
import numpy as np
import argparse
from PIL import Image
#from matplotlib.pyplot import imshow


def injection_effect(im_arr):
    """
    return a numpy array representing an image, s.t.
    the dimensions are (H, W, 3) (height, width, (r,g,b))
    given a similar array. Each coordinate in the new image
    comes from one in the original one.
    """
    print('creating zeros of %s' % str(im_arr.shape))
    Y, X, C = im_arr.shape
    bounds = np.array([Y, X])
    temp = np.zeros(im_arr.shape)
    center = np.array([int(Y / 2), int(X / 2)])
    print('center is at %s' % str(center))

    def valid_index(floating):
        return int(floating.clip(0, Y - 1))

    for y, x in np.ndindex(im_arr.shape[0:2]):
        recen = np.array([y, x]) - center
        recen_norm = recen / bounds * 2
        y_src, x_src = (center + (recen_norm ** 2) * recen)
        temp[y, x, :] = im_arr[valid_index(y_src), valid_index(x_src), :]
    return temp


if __name__ == "__main__":
    description = 'produce distorted versions of input files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='input files')

    for file in parser.parse_args().files:
        im = Image.open(file, 'r')
        im_arr = np.asarray(im)
        new_im_arr = injection_effect(im_arr)
        result_img = Image.fromarray(np.uint8(new_im_arr))
        new_file = file.replace('.jpg', '_sqr.jpg')
        print('saving new image to %s' % (new_file))
        result_img.save(new_file)
