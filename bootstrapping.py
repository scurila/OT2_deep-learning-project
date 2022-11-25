import cv2
import os
import random
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from net import Net
import torch.nn as nn
import torch.optim as optim


train_dir = './train_images'
# test_dir = './test_images'

transform = transforms.Compose(
    [transforms.Grayscale(), 
     transforms.ToTensor(), 
     transforms.Normalize(mean=(0,),std=(1,))
     ])

train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
# test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

batch_size = 32

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

# train_face_idx = random_idx_face[valid_size:]
# train_noface_idx = random_idx_noface[valid_size:]

# train_new_idx = train_face_idx + train_noface_idx

# train_sampler = SubsetRandomSampler(train_new_idx)
# train_sampler = ImbalancedDatasetSampler(train_data, indices=train_new_idx)
sampler = SubsetRandomSampler(list(range(len(train_data))))
train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, batch_size=batch_size, num_workers=1)

# 2nd step
bootsIterations = 6
thrFa = 0.8
n_epochs = 60

# Hyperparameters
lr = 0.01
momentum = 0.9

# Model choices
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = torch.nn.CrossEntropyLoss()

# sampler_type = type(train_sampler).__name__
# print(f'Training the NN (using {sampler_type}) with\nN_EPOCHS = {n_epochs}   lr = {lr}   momentum = {momentum} ...')

#3rd step

for bootsIter in range(bootsIterations):

    

    for epoch in range(1, n_epochs+1):  # loop over the dataset multiple times

        running_loss = 0.0
        if __name__ == '__main__': 
            for i, (data, target) in enumerate(train_loader):
                print(i)
                # zero the parameter gradients
                optimizer.zero_grad()
                print(i)
                # forward + backward + optimize
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                # Print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # Print every 2000 mini-batches
                    print(f'[epoch {epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0