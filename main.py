from functions import *
from scipy.optimize import curve_fit

print("loading the video...")
cap = cv2.VideoCapture("video_test.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("video_test.mp4")


print("computing hashes...")
L = []

while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2cvtColor(frame, cv2.COLOR_BGR2GRAY)
    L.append(phash64(gray))
    print(phash64(gray))

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

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

nb_bins = max(d.keys())
h, _, _ = plt.hist(LL, bins=nb_bins)

popt, pcov = curve_fit(func, np.array(list(range(1, nb_bins + 1))), h)


