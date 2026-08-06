import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def train(args, net, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    for epoch in range(args.num_epochs):
        net.train()
        running_loss, total, correct = 0.0, 0, 0
        for images, labels in tqdm(trainloader, desc='Epoch ' + str(epoch), unit='b'):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        print('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (
            epoch, loss, accuracy))


def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for images, labels in tqdm(testloader, desc='Test', unit='b'):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %.2f%%' % (
        100 * correct / total))


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
