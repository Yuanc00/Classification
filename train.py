from tqdm import tqdm

from dataset import ClassficationDataset
from models.LeNet import LeNet
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


def train():
    epochs = 5
    batch_size = 128
    device = torch.device('cpu')
    net = LeNet(10)
    print(net)
    net.to(device)

    transform = [transforms.Resize([64, 64]),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    dataset = ClassficationDataset('data/cifar/train', transform=transforms.Compose(transform), cache_img=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    dataset_test = ClassficationDataset('data/cifar/test', transform=transforms.Compose(transform), cache_img=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=8, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(0, epochs):
        running_loss = 0.0
        s = 'epoch ' + str(epoch + 1) + '/' + str(epochs) + ':' \
            ' ' % 48

        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=len(dataloader), desc=s)
        for i, (img, label) in pbar:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('loss:' + str(running_loss / len(dataloader)))
        # test(net, dataloader_test, device)

    print('Finished Training')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    train()