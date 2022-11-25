from scale_reduction import *
from PIL import Image
import os
import torch
from net import *
import torchvision
from nms import *

image = cv2.imread('family-portrait-half.jpg', cv2.IMREAD_GRAYSCALE)
image = image / 255.0 # Normalizing the image pixels because the model was trained with normalized values

# Size window
winW = 36
winH = 36

confidence_required = 0.995
stepSize = 6 # Pixel step for each window

def window_scale_reduction(net, winW, winH, image):
	# Remove all the contents of the cropped folder
	dir = 'cropped'
	for f in os.listdir(dir):
		os.remove(os.path.join(dir, f))

	img_nb = 0 # index of the cropped image

	all_faces = []
	faces_tensors = []
	rectangles = []
	# loop over the image pyramid
	for resized in pyramid(image, scale=1.25):
		faces = []
		curr_scale_factor = image.shape[0] / resized.shape[0]
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			
			# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
			# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
			# WINDOW
			window_tensor = torch.from_numpy(window)
			
			# resize tensor from the shape [36, 36] to [1,1,36,36]
			window_tensor = window_tensor[None, None, :, :]
			output = net(window_tensor.float())
			_, predicted = torch.max(output.data, 1)
			clone = resized.copy()
			if (predicted == torch.tensor([1])):
				m = torch.nn.Softmax(dim=1)
				face_prob = float(m(output)[0][1])
				# Select the crops that produce the highest confidence (probability) of it being a face
				if face_prob > confidence_required:
					faces.append((x, y, m(output)))
					crop_img = clone[y:y + winH, x:x + winW]
					crop_img = crop_img * 255.0
					cv2.imwrite('cropped/img-cropped-'+ str(img_nb)+'.jpg', crop_img)
					new_x = int(x * curr_scale_factor)
					new_y = int(y * curr_scale_factor)
					new_winW = int(winW * curr_scale_factor)
					new_winH = int(winH * curr_scale_factor)
					rectangles.append([(new_x, new_y), ((new_x + new_winW), (new_y + new_winH))])
					faces_tensors.append([new_x, new_y, new_x + new_winW, new_y + new_winH, face_prob])
					img_nb += 1

			# save the cropped image
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			# time.sleep(0.025)
			# img_nb += 1

		# Add the detected faces and the corresponding factors to the all_faces variable
		all_faces.append([curr_scale_factor, faces])

	# save the faces in a file
	with open('detected_faces.txt', 'w') as fp:
		for face in all_faces:
			# Write each item on a new line
			fp.write("%s\n" % face)

	# Apply nms
	tensor_faces = nms_pytorch(torch.tensor(faces_tensors), 0.2)
	#change 
	new_tensor_faces = []
	for face in tensor_faces:
		tuple1 = (int(face[0]),int(face[1]))
		tuple2 = (int(face[2]),int(face[3]))
		new_tensor_faces.append([tuple1,tuple2])


	return all_faces, rectangles,new_tensor_faces


def save_final_image(rectangles):
    image = cv2.imread('family-portrait-half.jpg')
    for rectangle in rectangles:
        cv2.rectangle(image, rectangle[0], rectangle[1], (0, 0, 255))
    cv2.imwrite('cropped/img-cropped-'+ 'final' + '.jpg', image)

def save_final_image_tensor(tensor):
    image = cv2.imread('family-portrait-half.jpg')
    for face in tensor:
        cv2.rectangle(image, face[0], face[1], (0, 0, 255))
    cv2.imwrite('cropped/img-cropped-'+ 'final_tensor' + '.jpg', image)

if __name__ == "__main__":
	net = Net()
	net.load_state_dict(torch.load("./models/net_11.pth"))
	all_faces, rectangles ,tensor_faces = window_scale_reduction(net, winW, winH, image)
	save_final_image(rectangles)
	#print(len(tensor_faces))
	print("tensors = ",tensor_faces[0])
	print("rectangele = ",rectangles[0])

	save_final_image_tensor(tensor_faces)