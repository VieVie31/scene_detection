from sys import argv, exit
from pprint import pformat, pprint
from os.path import abspath
from tqdm import tqdm, trange
from itertools import combinations
from collections import Counter
from functions import phash64, dhash, dhash_freesize, \
                      hamming, longuest_repeated_string, \
                      get_indexes, hamming_match, \
                      get_hash_of_hashes, sliding_window, \
                      compress_indexes, merge_intervals, mseq, cseq, \
                      get_scenes_segmentation, get_joly_scenes_sementation
from skvideo.io import vreader, ffprobe
from subprocess import call
from time import sleep
from base64 import b64decode
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation
import os
import warnings
import numpy as np

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
        frames = []

        # Hashing
        for _, frame in zip(trange(VIDEO_FRAMES_COUNT, leave=False), cap):
            hash_img = int(phash64(frame))
            if len(L):
                diffs.append(hamming(hash_img, L[-1]))
            L.append(hash_img)
            frames.append(frame)

        # Get scene segmentation and their hash
        scenes = get_joly_scenes_sementation(frames, nb_std_above_mean_th=2.)#get_scenes_segmentation(diffs, nb_std_above_mean_th=2.5)
        del frames #not take to much memory too long for nothing...
        scenes_hashes = [get_hash_of_hashes(L[s:e]) for s, e in scenes]
        
        #tqdm.write(pformat(Counter(scenes_hashes)))

        distance_matrix = np.zeros([len(scenes_hashes)] * 2)
        #compute distance between scenes' hashes
        for i in trange(len(scenes_hashes)):
            for j in range(len(scenes_hashes)):
                distance_matrix[i, j] = hamming(scenes_hashes[i], scenes_hashes[j])
        #find similar scenes which have hases distance too close compared to others
        similar_scenes_matrix = distance_matrix < 64 - (distance_matrix.mean() + distance_matrix.std() * 3)
        #try to automatically found clusters with affinity propagation
        cluster_builder = MeanShift(bandwidth=1)
        scenes_clusters = cluster_builder.fit_predict(similar_scenes_matrix)
        #find the clusters with 'too much' points inside compared to others
        clusters_counter = Counter(scenes_clusters)
        clusters_freq = np.array(list(clusters_counter.values()))
        clusters_freq_th = clusters_freq.mean() + clusters_freq.std() * 2.5
        frequent_clusters_id = list(filter(lambda k: clusters_counter[k] > clusters_freq_th, clusters_counter))
        #find hashes corresponding to these clusters
        scenes_hashes_idx = np.array(list(map(lambda v: v in frequent_clusters_id, scenes_clusters)))
        generics_scenes_hashes = np.array(scenes_hashes, dtype=np.uint64)[scenes_hashes_idx]
        #get the generics indexes from the scene hashes
        generics_scenes_idx = []
        for i, h in enumerate(scenes_hashes):
            if h in generics_scenes_hashes:
                generics_scenes_idx.append(i)
        #get the boundaries of gnerics scenes
        generics_scenes = list(map(lambda i: scenes[i], generics_scenes_idx))

        tqdm.write(str(len(generics_scenes)))
        tqdm.write(pformat(list(generics_scenes)))

        scenes_hashes_vector = list(map(lambda k: list(map(int, list(bin(k)[2:]))), scenes_hashes))
        scenes_hashes_vector = list(map(lambda v: [0] * (64 - len(v)) + v, scenes_hashes_vector)) # 0 padding to get the same length
        scenes_hashes_vector = np.array(scenes_hashes_vector)
        #TODO: make clustering to find clusters with the greatest density
        #those could be the generics and redundant parts...        
        kmean = KMeans(n_clusters=len(scenes_hashes_vector) // int(np.log(len(scenes_hashes_vector))))
        scenes_clusters = kmean.fit_predict(scenes_hashes_vector)
        scenes_clusters_freq = Counter(scenes_clusters)
        scene_freq = np.array(list(scenes_clusters_freq.values()))
        min_scene_freq = scene_freq.mean() + scene_freq.std() * 3
        redundant_scenes_clusters_id = list(filter(
            lambda k: scenes_clusters_freq[k] >= min_scene_freq,
            scenes_clusters_freq.keys()
        ))
        redudant_scenes = set([i if w else -1 for i, w in enumerate(map(lambda v: v in redundant_scenes_clusters_id, scenes_clusters))])
        try:
            redudant_scenes.remove(-1)
        except:
            pass
        redudant_scenes = sorted(list(redudant_scenes))
        print(len(redudant_scenes))

        for rs in redudant_scenes:
            print(scenes[rs])
        
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


        #for v in L:
        #    print(v)

        f = open("/cache/L.txt", "w")
        f.write(str(L))
        f.close()

        #try to automatically find a good SEQUENCE_LENGTH
        s_mseq = mseq(L)
        h_mseq = get_hash_of_hashes(s_mseq)
        print("potential sequence of ints representing the generic : ")
        print(s_mseq)
        print("correspinding sequence hash :")
        print(h_mseq)
        print(h_mseq)
        print(h_mseq)

        SEQUENCE_LENGTH = len(s_mseq)
        
        #trying some stuffs about sequence hashing
        sequences = list(sliding_window(L, SEQUENCE_LENGTH, SLIDING_WINDOW, get_hash_of_hashes))
        c = Counter(sequences)
        # tqdm.write(pformat(c, indent=2, depth=2))

        tqdm.write(pformat(c.most_common()[:10]))

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

    ###########
    indexes = np.array(generics_scenes).T[0]
    ###########
    
    tqdm.write(str(indexes))
    tqdm.write("# matchs : {}".format(len(indexes)))
    #"""
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
    #"""
