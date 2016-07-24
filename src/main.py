from functions import dhash, hamming, dhash_freesize
from os.path import abspath
from tqdm import tqdm, trange
from sys import argv
from collections import defaultdict
from pprint import pformat
from itertools import combinations
from skvideo.io import ffprobe, vreader
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
        diffs = []

        # Hashing
        count_tab = defaultdict(lambda: 0, {})
        for _, frame in zip(trange(VIDEO_FRAMES_COUNT, leave=False), cap):
            hash_img = int(dhash_freesize(frame))
            count_tab[hash_img] += 1
            if len(L):
                diffs += [hamming(hash_img, L[-1])]
            else:
                diffs += [0]
            L.append(hash_img)

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
        plotfile = '/videos/%s.png' % source.split('/')[-1]
        plt.savefig(plotfile)
    tqdm.write(pformat(stats, indent=2, depth=2))
