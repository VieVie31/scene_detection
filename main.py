from functions import *
from scipy.optimize import curve_fit

print("loading the video...")
cap = cv2.VideoCapture("video_test.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("video_test.mp4")


print("computing hashes...")
L = []
pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
    if flag: # The frame is ready and already captured
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        L.append(phash64(frame))
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)

    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break


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

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

nb_bins = max(d.keys())
h, _, _ = plt.hist(LL, bins=nb_bins)

popt, pcov = curve_fit(func, np.array(list(range(1, nb_bins + 1))), h)


