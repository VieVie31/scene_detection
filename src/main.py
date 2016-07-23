from functions import dhash
from os.path import abspath
from tqdm import tqdm
from sys import argv

import skvideo.io

if __name__ == '__main__':
    # Temporary arg parsing
    for source in argv[1:]:
        print('Hashing', source)
        VIDEO_PATH = abspath(source)
        cap = skvideo.io.vreader(VIDEO_PATH)
        infos = skvideo.io.vread(VIDEO_PATH, as_grey=True)

        print("computing hashes...")
        L = []

        with tqdm(total=len(infos)) as pbar:
            for frame in cap:
                phash = dhash(frame)
                #print(phash)
                L.append(phash)
                pbar.update()

        print(len(L))

