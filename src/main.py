from sys import argv
from pprint import pformat
from os.path import abspath
from tqdm import tqdm, trange
from itertools import combinations
from collections import Counter
from skvideo.io import ffprobe, vreader
from functions import phash64, dhash, dhash_freesize, hamming, longuest_repeated_string, get_indexes, hamming_match

import os
import warnings
# Remove matplotlib warning because of fc-cache building
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

if __name__ == '__main__':
    stats = []
    # Temporary arg parsing
    for _, source in zip(trange( \
                            len(argv) - 1, \
                            desc='Files', \
                            ), \
                         argv[1:]):

        # Loading source video
        VIDEO_PATH = abspath(source)
        METADATA = ffprobe(source)['video']
        VIDEO_FRAMES_COUNT = int(METADATA["@nb_frames"])
        cap = vreader(VIDEO_PATH)

        L = []
        diffs = [0]

        # Hashing
        for _, frame in zip(trange(VIDEO_FRAMES_COUNT, leave=False), cap):
            hash_img = int(phash64(frame))
            if len(L):
                diffs.append(hamming(hash_img, L[-1]))
            L.append(hash_img)

        count_tab = Counter(L)
        collisions = sum(count_tab.values()) - len(count_tab)
        assert len(L) == VIDEO_FRAMES_COUNT
        stats += [{
            'source': VIDEO_PATH,
            'frames': VIDEO_FRAMES_COUNT,
            'collisions': collisions,
            'datas': {
                'diffs': diffs,
                'hashes': L
            }
        }]

        # Creating output graph
        plt.plot(L)
        plotfile = '/cache/stats/%s.png' % source.split('/')[-1]
        plt.savefig(plotfile)
        plt.clf()
        plt.cla()
        plt.close()

        #searching the longuest repeated sub array
        #<!> this function take a quadratic time ... so can crash computer ?? maybe...
        potentiel_generic = longuest_repeated_string(L, lambda a, b: hamming(a, b) < 1)
        tqdm.write(pformat(potentiel_generic, indent=2))
        tqdm.write(str(len(potentiel_generic)))
        tqdm.write(str(get_indexes(L, potentiel_generic, lambda a, b: hamming_match(a, b, 5))))
    tqdm.write(pformat(stats, indent=2, depth=2))
