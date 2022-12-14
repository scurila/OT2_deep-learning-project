import imutils
import cv2
import os
import torch
from net import *
from nms import *

image_path = './fam_pictures/test_fam2.jpg'
model_path = './models/bootstrap/3ep-iter-2.pth'

winW = 36 # window width
winH = 36 # window height

threshold_nms = 0.2

confidence_required = 0.994 # threshold for the probability of a detected face
stepSize = 6 # pixel step for each window

# Images pyramid algorithm
def pyramid(image, scale=1.25, minSize=(30, 30)):
    	
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

# Sliding window algorithm
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# Apply pyramid and sliding window algorithms
def apply_sliding_window_image_piramid(net, winW, winH, image):
	# Remove all the contents of the 'cropped' folder
	dir = 'cropped'
	for f in os.listdir(dir):
		os.remove(os.path.join(dir, f))

	img_nb = 0 # index of the cropped image
	faces_positions = [] # detected faces positions in the original image
	faces_positions_tensor = [] # detacted faces positions in the original image and their probabiities
	# Loop over the image pyramid
	for resized in pyramid(image, scale=1.25):
		faces = []
		scale_value = image.shape[0] / resized.shape[0]
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			
			# Use the contents of the window as input of the neural network 
			window_tensor = torch.from_numpy(window)
			window_tensor = window_tensor[None, None, :, :] # resize tensor from the shape [36, 36] to [1,1,36,36]
			output = net(window_tensor.float())
			_, predicted = torch.max(output.data, 1)
			clone = resized.copy()
			# if a face is detected
			if (predicted == torch.tensor([1])):
				m = torch.nn.Softmax(dim=1)
				face_prob = float(m(output)[0][1]) # face probability
				# select the crops that produce the highest confidence (probability) of it being a face
				if face_prob > confidence_required:
					faces.append((x, y, m(output)))
					# save cropped detected face image
					crop_img = clone[y:y + winH, x:x + winW]
					crop_img = crop_img * 255.0
					cv2.imwrite('cropped/img-cropped-'+ str(img_nb)+'.jpg', crop_img)
					# detect x,y positions in the original size (before rescale) of the image and save them
					new_x = int(x * scale_value)
					new_y = int(y * scale_value)
					new_winW = int(winW * scale_value)
					new_winH = int(winH * scale_value)
					faces_positions.append([(new_x, new_y), ((new_x + new_winW), (new_y + new_winH))])
					faces_positions_tensor.append([new_x, new_y, new_x + new_winW, new_y + new_winH, face_prob])
					img_nb += 1

			# show the window on the resized image
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)

	# Apply nms
	new_faces = nms_pytorch(torch.tensor(faces_positions_tensor), threshold_nms)
	# save the positions of the detected faces as a list of tuples 
	new_faces_postions = []
	for face in new_faces:
		tuple1 = (int(face[0]),int(face[1]))
		tuple2 = (int(face[2]),int(face[3]))
		new_faces_postions.append([tuple1,tuple2])

	return faces_positions, new_faces_postions

# Show all detected faces with a red rectangle in the final image (isNms = true if we save it for the nms version)
def save_final_image(faces_positions, image_path, isNms):
    image = cv2.imread(image_path)
    for pos in faces_positions:
        cv2.rectangle(image, pos[0], pos[1], (0, 0, 255))
    if(isNms):
        cv2.imwrite('cropped/img-cropped-'+ 'final_nms' + '.jpg', image)
    else:			
    	cv2.imwrite('cropped/img-cropped-'+ 'final' + '.jpg', image)

if __name__ == "__main__":
	net = Net()
	net.load_state_dict(torch.load(model_path))
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	image = image / 255.0 # normalizing the image because the model was trained with normalized values
	faces_positions, new_faces_positions = apply_sliding_window_image_piramid(net, winW, winH, image)
	save_final_image(faces_positions, image_path, False) # save the detected faces before nms on a final image
	save_final_image(new_faces_positions, image_path, True)  # save the detected faces after nms on a final image
		