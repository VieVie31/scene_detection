#this file is not a module to import but contains some tests
#about the pattern discovery functions created for the scd project
if __name__ != "__main__":
    exit(42)

import math
import random
import argparse
import numpy as np

#import warnings
#remove matplotlib warning because of fc-cache building
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    import matplotlib.mlab as mlab
#    import matplotlib.pyplot as plt

from collections import Counter
from pprint import pprint
from sys import argv

#importing my set of functions
from functions import *

#create a parser to automatise some tests
parser = argparse.ArgumentParser(
        prog="mseq", 
        description='Test file used to benchmark different method for the generic finding problem.')
#add command to the parser
parser.add_argument('-bm',  '--bound-max',       type=int, help='The random numbers will uniformly be selected between 0 and bound-max.', default=2**8)
parser.add_argument('-tl',  '--total-length',    type=int, help='The length of the list containing random numbers.', default=10000)
parser.add_argument('-sl',  '--sequence-length', type=int, help='The length of the random pattern to insert.', default=10)
parser.add_argument('-ni',  '--nb-insert',       type=int, help='How many time the random pattern gotta be inserted randomly in the long initial sequence.', default=12)
parser.add_argument('-std', '--nb-std',          type=int, help='The number of standard deviation to consider a tri-gram could be part of the pattern.', default=4)

#parse the arguments
args = parser.parse_args(argv[1:])

BOUND_MAX = args.bound_max
L_LENGTH  = args.total_length
SL_LENGTH = args.sequence_length
NB_INSERT = args.nb_insert
NB_STD    = args.nb_std

print("bound_max,total_length,sequence_length,nb_insert,nb_std,sequence_length_lzw,hamming_distance_lzw_origin,sequence_length_stat,hamming_distance_stat_origin")

#generating the random sequence and pattern
L   = [random.randrange(0, BOUND_MAX) for i in range(L_LENGTH)]  #the long sequence
SL  = [random.randrange(0, BOUND_MAX) for i in range(SL_LENGTH)] #the shorter sequence
idx = [random.randrange(0, L_LENGTH) for i in range(NB_INSERT)]  #index in the lonsequence to appen the shorter
idx.sort()

#insert the pattern
for i in idx:
    L = L[:i] + SL + L[i:]

##f = open("/Users/mac/Desktop/scene_detection/cache/L.txt", 'r')
##L = eval(f.read())
##f.close()

print("{},{},{},{},{},".format(
    BOUND_MAX,
    L_LENGTH,
    SL_LENGTH,
    SL_LENGTH,
    SL_LENGTH), end='')

#compute the hash of the inserted sequence
h_o_h = get_hash_of_hashes(SL)


################################
# THE COMPRESSION LZW APPROACH #
################################

def index_of_sublist_in_list(lst, sub_lst):
    for i in range(len(lst) - len(sub_lst)):
        if lst[i:i+len(sub_lst)] == sub_lst:
            return i
    raise ValueError("Sub list not found !! :'(")

reconstitued_sequence = cseq(L)

print("{},{},".format(
    len(reconstitued_sequence),
    hamming(h_o_h, get_hash_of_hashes(reconstitued_sequence))), end='')

############################
# THE STATISTICAL APPROCAH #
############################

reconstitued_sequence = mseq(L, nb_std=NB_STD)
print("{},{}".format(
    len(reconstitued_sequence),
    hamming(h_o_h, get_hash_of_hashes(reconstitued_sequence))), end='')

print('')



