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
#        VIDEO_FRAMES_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("computing hashes...")
        L = []

        with tqdm() as pbar:
            for frame in cap:
                phash = dhash(frame)
                print(phash)
                L.append(phash)
                pbar.update()

