import math
import random
import numpy as np

import warnings
#remove matplotlib warning because of fc-cache building
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

from collections import Counter
from pprint import pprint

#importing my set of functions
from functions import *


BOUND_MAX = 2**8
L_LENGTH = 10000
SL_LENGTH = 10
NB_INSERT = 20
NB_STD = 4

print("generation...")

L   = [random.randrange(0, BOUND_MAX) for i in range(L_LENGTH)]  #the long sequence
SL  = [random.randrange(0, BOUND_MAX) for i in range(SL_LENGTH)] #the shorter sequence
idx = [random.randrange(0, L_LENGTH) for i in range(NB_INSERT)]  #index in the lonsequence to appen the shorter
idx.sort()


for i in idx:
    L = L[:i] + SL + L[i:]

NB_STD = 4

##f = open("/Users/mac/Desktop/scene_detection/cache/L.txt", 'r')
##L = eval(f.read())
##f.close()

################################
# THE COMPRESSION LZW APPROACH #
################################

def index_of_sublist_in_list(lst, sub_lst):
    for i in range(len(lst) - len(sub_lst)):
        if lst[i:i+len(sub_lst)] == sub_lst:
            return i
    raise ValueError("Sub list not found !! :'(")

reconstitued_sequence = cseq(L)

print(reconstitued_sequence)
print("found exactly the same sequence : ", reconstitued_sequence == SL)

############################
# THE STATISTICAL APPROCAH #
############################

print("searching...")

print(len(L))

NB_STD = 4

d = {}
for i in range(len(L) - 2): #-1):#- 2):
    tpl = ((L[i], L[i + 1]), L[i + 2]) #(L[i], L[i + 1]) #((L[i], L[i + 1]), L[i + 2])
    if tpl in d:
        d[tpl] += 1
    else:
        d[tpl] = 1


a = list(sorted(list(d.values())))
a = np.array(a)


#keep number bigger than 6 standard deviation (cf. anomally detection)
#keep keys in d that have a counter bigger than NB_STD * std
dd = {}
for k in d:
    if d[k] >= a.mean() + NB_STD * a.std():
        dd[k] = d[k]

#remap and keep the keys
dd = {(a, b, c) : [] for ((a, b), c) in dd.keys()}

#create all possibles paths
for (a, b, c) in dd:
    dd[(a, b, c)] = filter(lambda t: (b, c) == (t[0], t[1]), dd.keys())

#get the longest path without cycle --> should be the SL (or close to, statistically)
longest = []

visited = set()
path = []
def search(v, d={}):
    visited.add(v)
    path.append(v)
    
    if len(path) > len(longest):
        longest[:] = path[:]
        
    for w in d[v]:
        if w not in visited:
            search(w, d)
            
    path.pop()
    visited.remove(v)

#try all possibilities
for v in dd:
    search(v, dd)

#remake the list of longest
L = list(longest[0])
for t in longest[1:]:
    L.append(t[2])

#print(L == SL)
print(L)
#print(SL)


