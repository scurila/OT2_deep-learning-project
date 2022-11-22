from scale_reduction import *
from PIL import Image
import os

image = cv2.imread('family-portrait.jpg', cv2.IMREAD_GRAYSCALE)
# size window
winW = 36
winH = 36

# Remove all the contents of the cropped folder
dir = 'cropped'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))


img_nb = 0 # index of the cropped image

# loop over the image pyramid
for resized in pyramid(image, scale=1.25):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=6, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		crop_img = clone[y:y + winH, x:x + winW]
		# print(crop_img)
		Image.fromarray(crop_img,mode='L').save('cropped/img-cropped-'+str(img_nb)+'.jpg')
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		#time.sleep(0.025)
		img_nb += 1