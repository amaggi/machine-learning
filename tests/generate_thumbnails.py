#! /usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def generate_thumbnails(number, filename, order='F'):
    plt.imshow(number.reshape((20,20), order=order))
    plt.savefig(filename)

    return filename

def filenames(numbers):
    for number in range(1, len(numbers)+1):
        yield "/tmp/numbers_%d.png" % number


if __name__ == '__main__':
    data = dict(np.load('data.npz').items())

    numbers = data['numbers']

    with ProcessPoolExecutor() as executor:
        for filename in executor.map(generate_thumbnails, numbers, filenames(numbers)):
            print(filename)
