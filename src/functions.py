#import cv2
import itertools
import numpy as np

from struct import pack
from skimage.color import rgb2grey
from skimage.transform import resize

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
    it = itertools.zip_longest(a, b, fillvalue='0')
    return sum([va != vb for (va, vb) in it])

def phash64(img):
    """Compute a perceptual hash of an image.

    :param img: a rgb image to be hashed

    :type img: numpy.ndarray

    :return: a perceptrual hash of img coded on 64 bits
    :rtype: int
    """
    resized = rgb2grey(resize(img, (8, 8)))
    mean = resized.mean()
    boolean_matrix = resized > mean
    hash_lst = boolean_matrix.reshape((1, 64))[0]
    hash_lst = list(map(int, hash_lst))
    im_hash = 0
    for v in hash_lst:
        im_hash  = (im_hash << 1) | v
    return im_hash

def phash1(img):
    """Return the hash of the image as a list of bits
    always ordered in the same order.

    :param img: the binarized image

    :type img: numpy.ndarray

    :return: the perceptual hash of img in a vector
    <!> the length of the hash returned depend of the size of image !!
    :ntype: np.array
    """
    return ((img > 0) * 1).reshape((1, img.shape[0] * img.shape[1]))[0]

def dhash(img):
    """Compute a perceptual has of an image.

    Algo explained here :
    https://blog.bearstech.com/2014/07/numpy-par-lexemple-une-implementation-de-dhash.html

    :param img: an image

    :type img: numpy.ndarray

    :return: a perceptual hash of img coded on 64 bits
    :rtype: int
    """
    TWOS = np.array([2 ** n for n in range(7, -1, -1)])
    BIGS = np.array([256 ** n for n in range(7, -1, -1)], dtype=np.uint64)
    img = rgb2grey(resize(img, (9, 8)))
    h = np.array([0] * 8, dtype=np.uint8)
    for i in range(8):
        h[i] = TWOS[img[i] > img[i + 1]].sum()
    return (BIGS * h).sum()

def dhash_freesize(img):
    bits = []
    for row in range(img.shape[0]):
        for col in range(img.shape[1]-1):
            l = pack('BBB', *img[row][col])
            r = pack('BBB', *img[row][col+1])
            bits.append('1') if l > r else bits.append('0')

    dhash = int(''.join(bits), 2)
    return dhash

def histogram(vector):
    """Compute the histogram of a vector.

    :param vector: a list of values

    :type vector: list

    :return: the histogram of values as a dictionnary
    :rtype: dict
    """
    return {k: vector.count(k) for k in set(vector)}

def variance(hist:dict):
    """Compute the variance of an histogram.

    :param hist: the histogram

    :type hist: dict

    :return: the variance of the histogram
    :rtype: float
    """
    vl = list(hist.values())
    m = sum(vl) / float(len(vl))
    return sum([(m - v)**2 for v in vl]) / float(len(vl))
