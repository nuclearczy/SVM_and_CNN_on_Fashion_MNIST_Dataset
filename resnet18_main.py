import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time
import matplotlib.pyplot as plt


def main():
    PRE_TRAINED = 1
    RESNET18_PATH = './weight/FashionMNIST_resnet18_20epoch.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    resnet18 = models.resnet18()
    resnet18.to(device)
    # alexnet = models.alexnet()
    # vgg16 = models.vgg16()
    # squeezenet = models.squeezenet1_0()
    # densenet = models.densenet161()
    # inception = models.inception_v3()
    # googlenet = models.googlenet()
    # shufflenet = models.shufflenet_v2_x1_0()
    # mobilenet = models.mobilenet_v2()
    # resnext50_32x4d = models.resnext50_32x4d()
    # wide_resnet50_2 = models.wide_resnet50_2()
    # mnasnet = models.mnasnet1_0()

    trainset_fashion = torchvision.datasets.FashionMNIST(
        root='./data/pytorch/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    testset_fashion = torchvision.datasets.FashionMNIST(
        root='./data/pytorch/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    trainloader_fashion = torch.utils.data.DataLoader(trainset_fashion, batch_size=4,
                                                      shuffle=True, num_workers=2)
    testloader_fashion = torch.utils.data.DataLoader(testset_fashion, batch_size=4,
                                                     shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
    if (PRE_TRAINED):
        resnet18.load_state_dict(torch.load(RESNET18_PATH))
    else:
        accuracy_list = []
        epoch_list = []
        start_time = time.time()
        print("Start Training >>>")
        for epoch in range(20):
            running_loss = 0.0
            for i, data in enumerate(trainloader_fashion, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.repeat(1, 3, 1, 1)
                optimizer.zero_grad()
                outputs = resnet18(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 2000}')
                    running_loss = 0.0

            start_test = time.time()
            print(f"\nStart Epoch {epoch + 1} Testing >>>")
            correct = 0
            total = 0
            with torch.no_grad():
                for i, data in enumerate(testloader_fashion, 0):
                    images, labels = data[0].to(device), data[1].to(device)
                    images = images.repeat(1, 3, 1, 1)
                    outputs = resnet18(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if i % 2000 == 1999:
                        print(f'Testing Batch: {i + 1}')
            test_time = (time.time() - start_test) / 60
            print('>>> Finished Testing')
            print(f'Testing time: {test_time} mins.')
            print(f'Epoch {epoch + 1} Accuracy: {100 * correct / total}')
            accuracy_list.append(100 * correct / total)
            epoch_list.append(epoch + 1)

        train_time = (time.time() - start_time) / 60
        torch.save(resnet18.state_dict(), RESNET18_PATH)
        print('>>> Finished Training')
        print(f'Training time: {train_time} mins.')

        plt.plot(epoch_list, accuracy_list, 'b--', label='ResNet18 Accuracy')
        plt.title('ResNet18 Accuracy vs epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        axes = plt.gca()
        axes.set_ylim([0, 100])
        plt.legend()
        plt.savefig('./visualization/Resnet18vsEpoch.png')
        plt.show()




    start_test = time.time()
    print("\nStart Final Testing >>>")
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader_fashion, 0):
            images, labels = data[0].to(device), data[1].to(device)
            images = images.repeat(1, 3, 1, 1)
            outputs = resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 2000 == 1999:
                print(f'Testing Batch: {i + 1}')
    test_time = (time.time() - start_test) / 60
    print('>>> Finished Testing')
    print(f'Testing time: {test_time} mins.')
    print(f'Final Accuracy: {100 * correct / total}')


if __name__ == '__main__':
    main()
