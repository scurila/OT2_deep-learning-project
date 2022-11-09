import torch

from net import Net
import torchvision
import torchvision.transforms as transforms

PATH = './architectures/net_9.pth'

net = Net()
net.load_state_dict(torch.load(PATH))

test_dir = './test_images'    # folder containing test images

transform = transforms.Compose(
    [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
     transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
     transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)
batch_size = 32    

test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

correct = 0
total = 0
i = 0 # number of iterations
print_every_n_batch = 100
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % print_every_n_batch == 0:
            print(i)
        i += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

