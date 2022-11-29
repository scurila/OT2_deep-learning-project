import cv2
import os
import random
import imutils
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from net import Net
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sliding_window import sliding_window

winW = 36
winH = 36

rescale_factor_bootstrap = 3
stepSize_bootstrap = 6

max_new_fa = 12845 # 1/5 of the OG faces

def bootstrapping():
    train_dir = '../CNN_project/train_images'
    false_alarm_dir = './false_alarms'
    # test_dir = './test_images'

    batch_size = 32
    n_epochs = 3

    # Hyperparameters
    lr = 0.01
    momentum = 0.9

    # Model choices
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [transforms.Grayscale(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0,),std=(1,))
        ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    # test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    # Take a fixed validation set
    number_of_faces = len([img for img in train_data.imgs if img[1] == 1])
    valid_size = int(np.floor(0.016 * number_of_faces)) # Taking 1.6% of faces

    num_train = len(train_data)
    number_of_nofaces = num_train - number_of_faces
    indices_train = list(range(num_train))

    random_idx_noface = indices_train[:number_of_nofaces]
    random_idx_face = indices_train[number_of_nofaces:]

    np.random.shuffle(random_idx_face)
    np.random.shuffle(random_idx_noface)

    valid_face_idx = random_idx_face[:valid_size]
    valid_noface_idx = random_idx_noface[:valid_size]
    valid_new_idx = valid_face_idx + valid_noface_idx

    valid_sampler = SubsetRandomSampler(valid_new_idx)
    # valid_sampler = ImbalancedDatasetSampler(train_data, indices=valid_new_idx)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)

    # Remove this from the train data

    imgs = [train_data.imgs[i] for i in valid_new_idx]

    for img in imgs:
        train_data.imgs.remove(img)

    # Validation done

    # 2nd step
    bootsIterations = 6
    thresholdFace = 0.8

    # sampler_type = type(train_sampler).__name__
    # print(f'Training the NN (using {sampler_type}) with\nN_EPOCHS = {n_epochs}   lr = {lr}   momentum = {momentum} ...')

    # 3rd step

    n_nofaces_og = len([img for img in train_data.imgs if img[1] == 0]) # Original number of non-faces (after removing validation set)

    num_train_og = len(train_data) # Original number of faces and non-faces (after removing validation set)

    n_faces_og = num_train_og - n_nofaces_og  # Original number of faces (after removing validation set)
    indices_train = list(range(num_train_og))

    # We decided we will use 1778 (random number) texture/scenery images for the false alarm examples

    for bootsIter in range(bootsIterations):

        # Grabbing balanced subsets of both faces and nofaces
        idx_noface = indices_train[:n_nofaces_og]
        idx_face = indices_train[n_nofaces_og:] # [6000, 6001, ... , 27 000]
        # Grabbing n_nofaces_og shuffled faces
        np.random.shuffle(idx_face)
        new_idx_face = idx_face[:n_nofaces_og]

        new_idx_train = idx_noface + new_idx_face

        train_sampler = SubsetRandomSampler(new_idx_train)

        # print(len(new_idx_train))
        # for idx in idx_noface:
        #     if train_data[idx][1] == 0:
        #         print('no')

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)

        # for epoch in range(n_epochs):  # loop over the dataset multiple times

        #     running_loss = 0.0
        #     for i, (data, target) in enumerate(train_loader):
        #         # zero the parameter gradients
        #         optimizer.zero_grad()
        #         # forward + backward + optimize
        #         outputs = model(data)
        #         loss = criterion(outputs, target)
        #         loss.backward()
        #         optimizer.step()
                
        #         # Print statistics
        #         running_loss += loss.item()
        #         if i % 1000 == 999:    # Print every 1000 mini-batches
        #             print(f'[epoch {epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #             running_loss = 0.0
        
        # Save the model 
        # PATH = './models/net_test_bootstrap3.pth'
        # torch.save(model.state_dict(), PATH)

        # Load the model
        PATH = './models/net_test_bootstrap3.pth'
        model = Net()
        model.load_state_dict(torch.load(PATH))


        # Finished training

        fa_count = 0

        for img_path in os.listdir('texture_imgs'):
            if fa_count == max_new_fa: # Check if the maximum n of false alarms has been reached
                break
            image = cv2.imread('./texture_imgs/' + img_path, cv2.IMREAD_GRAYSCALE)
            w_bootstrap = int(image.shape[1] / rescale_factor_bootstrap)
            resized = imutils.resize(image, width=w_bootstrap)

            for (x, y, window) in sliding_window(resized, stepSize=stepSize_bootstrap, windowSize=(winW, winH)):
                if fa_count == max_new_fa:
                    break
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                
                # Use the contents of the window as input of the neural network 
                window_tensor = torch.from_numpy(window)
                window_tensor = window_tensor[None, None, :, :] # resize tensor from the shape [36, 36] to [1,1,36,36]
                output = model(window_tensor.float())
                _, predicted = torch.max(output.data, 1)
                clone = resized.copy()
                if predicted == torch.tensor([1]):    
                    # save cropped detected face image
                    m = torch.nn.Softmax(dim=1)
                    face_prob = float(m(output)[0][1]) # face probability
                    if face_prob > thresholdFace:
                        crop_img = clone[y:y + winH, x:x + winW]
                        cv2.imwrite('false_alarms/0/fa-'+ str(fa_count)+'.jpg', crop_img)
                        fa_count += 1

                # # show the window on the crops_ image
                # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                # cv2.imshow("Window_boostrap", clone)
                # cv2.waitKey(1)

        # False alarms saved

        false_alarm_data = torchvision.datasets.ImageFolder(false_alarm_dir, transform=transform)

        false_alarm_count = len(false_alarm_data)
        print(false_alarm_count)

        concat = torch.utils.data.ConcatDataset([train_data, false_alarm_data]) # Append nofaces at the end


        # num_train = len(train_data)
        # indices_train = list(range(num_train))
        # np.random.shuffle(indices_train)
        # split_tv = int(np.floor(valid_size * num_train))
        # train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

        # train_loader

        # Training

if __name__ == '__main__':
    bootstrapping()