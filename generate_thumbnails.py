#! /usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import sys


def generate_thumbnails(number, filename, order='F'):
    plt.imshow(number.reshape((20, 20), order=order))
    plt.savefig(filename)

    return filename


def filenames(numbers, directory):
    for number in range(1, len(numbers)+1):
        path = os.path.join(directory, "numbers_%d.png" % number)
        yield path


if __name__ == '__main__':
    directory = sys.argv[1]

    data = dict(np.load('data.npz').items())

    numbers = data['numbers']

    with ProcessPoolExecutor() as executor:
        filenames = map(
            generate_thumbnails,
            numbers,
            filenames(numbers, directory)
        )

        for filename in filenames:
            print(filename)
