import torch

from net import Net
import torchvision
import torchvision.transforms as transforms

PATH = './models/bootstrap/test-iter-2.pth'

def test_model(model, test_loader):
    correct = 0
    total = 0
    i = 0 # number of iterations
    print_every_n_batch = 100

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
       

    classes = ('noface', 'face')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == '__main__':
    net = Net()
    net.load_state_dict(torch.load(PATH))

    test_dir = '../CNN_project/test_images'    # folder containing test images

    transform = transforms.Compose(
        [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
        transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
        transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)
    batch_size = 32    

    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    test_model(net, test_loader)

