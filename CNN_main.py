import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


def data_show(img):
    with torch.no_grad():
        img_cpu = img.cpu()
        npimg = img_cpu.numpy()
        for img_to_show in npimg:
            npdata = img_to_show[0]
            print("data shape: ", np.shape(npdata))
            plt.imshow(npdata)
            plt.savefig(f'visualization/CNN/{time.time()}.png')
            plt.show()


def imshow(img):
    with torch.no_grad():
        img_cpu = img.cpu()
        npimg = img_cpu[0].numpy()
        for img_to_show in npimg:
            print("img shape: ", np.shape(img_to_show))
            plt.imshow(img_to_show)
            plt.savefig(f'visualization/CNN/{time.time()}.png')
            plt.show()


class playground_net(nn.Module):
    def __init__(self):
        super(playground_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # data_show(x)
        x = self.pool(F.relu(self.conv1(x)))
        # imshow(x)
        x = self.pool(F.relu(self.conv2(x)))
        # imshow(x)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    PRE_TRAINED = 0
    NET_PATH = './weight/FashionMNIST_playground_net.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = playground_net()
    net.to(device)

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
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    if (PRE_TRAINED):
        net.load_state_dict(torch.load(NET_PATH))
    else:
        accuracy_list = []
        epoch_list = []
        start_time = time.time()
        print("Start Training >>>")
        for epoch in range(20):
            running_loss = 0.0
            for i, data in enumerate(trainloader_fashion, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
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
                    outputs = net(images)
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
        torch.save(net.state_dict(), NET_PATH)
        print('>>> Finished Training')
        print(f'Training time: {train_time} mins.')

        plt.plot(epoch_list, accuracy_list, 'b--', label='Custom CNN Accuracy')
        plt.title('Custom CNN Accuracy vs epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        axes = plt.gca()
        axes.set_ylim([0, 100])
        plt.legend()
        # plt.savefig('./visualization/CustomCNNvsEpoch.png')
        plt.show()


if __name__ == '__main__':
    main()
