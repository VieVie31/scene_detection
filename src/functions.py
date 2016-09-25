#import cv2
import itertools
import numpy as np

from struct import pack
from functools import reduce
from collections import Counter
from skimage.color import rgb2grey
from skimage.transform import resize


def merge_intervals(indexes, length):
    """Merges overlapping intervals of matches for given indexes and generic
    lzngth. This function assume the indexes are allready sorted in ascending
    order.

    :param indexes: the list of indexes acs sorted to merge
    :param length: the length of the generic

    :type indexes: list of int
    :type length: int

    :return: a list of couples of begining of the interval and the end
    :rtype: list of list
    """
    out = []
    L = list(map(lambda v: [v, v + length], indexes))
    tmp = L[0]
    for (s, e) in L: #allready sorted
        if s <= tmp[1]:
            tmp[1] = max(tmp[1], e)
        else:
            out.append(tuple(tmp))
            tmp = [s, e]
    out.append(tuple(tmp))
    return out

def compress_indexes(indexes):
    """Compress a list of indexes. The list is assumed to be sorted in ascending
    order, and this function will remove the all the consecutives numbers and
    only keep the first and the number of the consecutives in a dict.

    eg : [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22]
    become : {0: 5, 18: 5, 7: 7}

    :param indexes: the list of indexes asc sorted to compress

    :type indexes: list of int

    :return: a dict containing the indexes as keys and # of consecutives
             numbers as value.
    :rtype: dict
    """
    d = {}
    c = indexes[0]
    j = 0
    for i in range(1, len(indexes)):
        j += 1
        if c + j != indexes[i]:
            d[c] = j
            c = indexes[i]
            j = 0
    d[c] = j + 1
    return d

def get_indexes(lst, sub_lst, compare_function=None):
    """Return the indexes of a sub list in a list.

    :param lst: the list to search in
    :param sub_lst: the list to match
    :param compare_function: the comparaison function used

    :type lst: list
    :type sub_lst: list
    :type compare_function: function, takes 2 list as argument

    :return: the list of indexes of sub_lst in lst.
    :rtype: list of int
    """
    indexes = []
    ln = len(sub_lst)
    for i in range(len(lst)):
        if compare_function:
            if compare_function(lst[i:i + ln], sub_lst):
                indexes.append(i)
        else:
            if lst[i:i + ln] == sub_lst:
                indexes.append(i)
    return indexes

def longuest_common_prefix(s, t, compare_function=None):
    """Return the longest common prefix of s and t.
    
    :param s: a string
    :param t: a string
    :param compare_function: a function for comparing two items

    :type s: str
    :type t: str
    :type compare_function: function (musta take 2 args)

    :return: the longest common prefix of s and t
    :rtype: str
    """
    n = min(len(s), len(t))
    for i in range(n):
        if compare_function:
            if not compare_function(s[i], t[i]):
                return s[:i]
        else:
            if s[i] != t[i]:
                return s[:i]
    return s[:]

def longuest_repeated_string(s, compare_function=None):
    """Return the longest repeated string in s.

    :param s: the string to find the longuest repeated subtring
    :param compare_function: a function for comparing two items 

    :type s: str

    :return: the longest repeated string in s
    :rtype: str
    """
    #form the N suffixes
    n = len(s)
    suffixes = []
    for i in range(n):
        suffixes.append(s[i:])

    #sort them
    suffixes.sort()

    #find longest repeated substring by comparing
    #adjacent sorted suffixes
    lrs = ""
    for i in range(n - 1):
        x = longuest_common_prefix(suffixes[i], suffixes[i+1], compare_function)
        if len(x) > len(lrs):
            lrs = x

    return lrs

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

def hamming_match(L1, L2, tolerence=2):
    """Tell us is 2 list of ints are equals with an error tolerance
    using the hamming ditance between 2 ints.

    :param L1: a list to compare
    :param L2: the second list to compare
    :param tolerence: the distance accepted between 2 ints

    :type L1: list of int
    :type L2: list of int
    :type tolerence: int
    """
    if len(L1) != len(L2):
        return False
    for (u, v) in zip(L1, L2):
        if hamming(u, v) > tolerence:
            return False
    return True

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

def int_to_bits_indexes(n):
    """Return the list of bits indexes set to 1.

    :param n: the int to convert

    :type n: int

    :return: a list of the indexes of bites sets to 1
    :rtype: list
    """
    L = []
    i = 0
    while n:
        if n % 2:
            L.append(i)
        n //= 2
        i += 1
    return L

def get_hash_of_hashes(L):
    """Return a compressed perceptual hash from
    a list of perceptual hashes.

    :param L: as list of hashes

    :type L: list of int

    :return: a compressed perceptual hash of a sequence
    :rtype: int
    """
    L = reduce(lambda a, b: a + b, map(int_to_bits_indexes, L), [])
    c = Counter(L)
    out = 0
    if c == Counter(): # a fix, because not working if no arg is passed to scd on os x
        return 0
    med = sum(c.values()) / (max(c) + 1.)
    for k in c:
        if c[k] >= med:
            out += 2 ** k
    return out

def sliding_window(L, slice_size, step=1, function=lambda v: v):
    """Iterator, apply some function on slices of 1D list,
    like a sliding window.

    :param L: list to apply the function
    :param slice_size: max size of the slice to apply the function
    :param step: step for forwarding, (overlaping the windows)
    :param function: the function to apply on the window

    :type L: list
    :type slice_size: int
    :type step: int
    :type function: function

    :yield: what the function return
    """
    i = 0
    while i < len(L):
        yield function(L[i:i+slice_size])
        i += step

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
