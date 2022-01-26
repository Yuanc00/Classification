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


if __name__ == "__main__":
    train()