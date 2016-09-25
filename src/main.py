from sys import argv, exit
from pprint import pformat
from os.path import abspath
from tqdm import tqdm, trange
from itertools import combinations
from collections import Counter
from functions import phash64, dhash, dhash_freesize, \
                      hamming, longuest_repeated_string, \
                      get_indexes, hamming_match, \
                      get_hash_of_hashes, sliding_window, \
                      compress_indexes, merge_intervals
from skvideo.io import vreader, ffprobe
from subprocess import call
from time import sleep
from base64 import b64decode
import os
import warnings

def get_metdata(path: str):
    a = ffprobe(path)['video']
    # print('FFPROBE :', path, pformat(a))
    return a

if __name__ == '__main__':
    stats = []

    SLIDING_WINDOW  = int(os.environ['SLIDING_WINDOW'])
    SEQUENCE_LENGTH = int(os.environ['SEQUENCE_LENGTH'])

    ORIGINAL_VIDEOS = []
    # First argument is a path to original videos informations
    with open(argv[1], 'r') as f:
        for line in f:
            idx, source, encoded = line.split(';')
            ORIGINAL_VIDEOS += [{
                "index": int(idx) - 1,
                "source": source,
                "encoded": encoded.replace('\n', '')
            }]
        ORIGINAL_VIDEOS = [(v['source'], v['encoded']) for v in sorted(ORIGINAL_VIDEOS, key=lambda x: x['index'])]
        print(ORIGINAL_VIDEOS)
        
    # Temporary arg parsing
    for _, source in zip(trange( \
                            len(argv) - 2, \
                            desc='Files', \
                            ), \
                         argv[2:]):

        # Loading source video
        VIDEO_PATH = abspath(source)

        if not os.access(VIDEO_PATH, os.R_OK):
            print('Internal error, source file is not avaliable')
            exit(1)
        METADATA = get_metdata(source)
        VIDEO_FRAMES_COUNT = int(METADATA["@nb_frames"])
        VIDEO_TIME_BASE = eval(METADATA["@codec_time_base"])
        # VIDEO_FRAMES_COUNT = 100000
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
            'time_base': VIDEO_TIME_BASE,
            'collisions': collisions,
            'datas': {
                'diffs': diffs,
                'hashes': L
            },
        }]


        #trying some stuffs about sequence hashing
        sequences = list(sliding_window(L, SEQUENCE_LENGTH, SLIDING_WINDOW, get_hash_of_hashes))
        c = Counter(sequences)
        # tqdm.write(pformat(c, indent=2, depth=2))


        best = c.most_common(1)[0][0]
        indexes = [x for x, s in enumerate(sequences) if s == best]
        #tqdm.write('[matchs] %d [indexes] %s' % (
        #    c.most_common(1)[0][1],
        #    indexes
        #))

    tqdm.write(pformat(stats, indent=2, depth=2))
    tqdm.write(pformat(ORIGINAL_VIDEOS, indent=2, depth=2))
    for stat in stats:
        tqdm.write('Frames: %s %s %s' % (stat['frames'], 'TimeBase:', stat['time_base']))
    start_frame = 0
    idx = 0


    ## DEPRECATED
    ##compressed_indexes = compress_indexes(indexes)
    ## we can choose to take juste the first index matched by sequence
    ## or take the middle sequence index... just a question of POV...
    ##indexes = d.keys() #here we take the first
    ##indexes = map(lambda k: int(k + compressed_indexes[k] / 2.),
    ##              compressed_indexes.keys()) #here we take the mean
    indexes = map(lambda t: (t[0] + t[1]) // 2,
                  merge_intervals(indexes, SEQUENCE_LENGTH))

    indexes = sorted(indexes)

    tqdm.write(str(indexes))
    tqdm.write("# matchs : {}".format(len(indexes)))
    """
    for match in indexes:
        found = False

        for source, dest in ORIGINAL_VIDEOS[idx:]:
            # destination video
            time_base = eval(get_metdata(dest + ".mp4")["@codec_time_base"])
            video_frames = int(get_metdata(dest + ".mp4")["@nb_frames"])

            # print(start_frame, '+', video_frames, '>', match)
            if start_frame + video_frames > match:
                print('timeout 2 vlc --start-time=%s "videos/%s";' % (match - start_frame, source))
                found = True
                break
            idx += 1
            start_frame += video_frames

        # If this assertion raise, we were not able to locate match in source videos
        assert found is True
    """
