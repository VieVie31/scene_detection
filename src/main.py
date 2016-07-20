from functions import *
from scipy.optimize import curve_fit
from os.path import abspath
from tqdm import tqdm

VIDEO_PATH = abspath("videos/video_test.mp4")
print("[INIT] Loading video at", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
while not cap.isOpened():
    cap = cv2.VideoCapture(VIDEO_PATH)

VIDEO_FRAMES_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("computing hashes...")
L = []

with tqdm(total=VIDEO_FRAMES_COUNT) as pbar:
    while (cap.isOpened()):
        ret, frame = cap.read(0) # 0 flag for an already greyscaled image
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        phash = phash64(frame)
        L.append(phash)
        pbar.update()

cap.release()

exit()

print("computing inter frames difference...")
LL = [hamming(L[i - 1], L[i]) for i in range(1, len(L))]

d = histogram(LL)
sigma = variance(d)**.5

#display the distance between 2 frames during the video
plt.plot(LL)
plt.show()

#display the repartition of inter frames distance
plt.hist(LL, bins=max(d.keys()))
plt.show()

nb_bins = max(d.keys())
h, _, _ = plt.hist(LL, bins=nb_bins)

popt, pcov = curve_fit(lambda x, a, b, c: a * np.exp(-b * x) + c, \
                np.array(list(range(1, nb_bins + 1))), h)
