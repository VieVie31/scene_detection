import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

def hamming(a, b):
    """Compute the hamming distance between 2 int.

    :param a: a 64 bits integer
    :param b: a 64 bits integer
    
    :type a: int
    :type b: int

    :return: the hamming distance between a, b
    :rtype: int
    """
    a = bin(a)[2:][::-1]
    b = bin(b)[2:][::-1]
    it = itertools.izip_longest(a, b, fillvalue='0')
    return sum([va != vb for (va, vb) in it])


def phash64(img):
    """Compute a perceptual hash of an image.

    :param img: a rgb image to be hashed

    :type img: numpy.ndarray

    :return: a perceptrual hash of img coded on 64 bits
    :rtype: int
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (8, 8))
    mean = cv2.mean(resized)[0]
    boolean_matrix = resized > mean
    hash_lst = boolean_matrix.reshape((1, 64))[0]
    hash_lst = map(int, hash_lst)
    im_hash = 0
    for v in hash_lst:
        im_hash  = (im_hash << 1) | v
    return im_hash

def histogram(vector):
    """Compute the histogram of a vector.

    :param vector: a list of values

    :type vector: list

    :return: the histogram of values as a dictionnary
    :rtype: dict
    """
    return {k: vector.count(k) for k in set(vector)}

def variance(hist):
    """Compute the variance of an histogram.

    :param hist: the histogram

    :type hist: dict

    :return: the variance of the histogram
    :rtype: float
    """
    vl = hist.values()
    m = sum(vl) / float(len(vl))
    return sum([(m - v)**2 for v in vl]) / float(len(vl))
