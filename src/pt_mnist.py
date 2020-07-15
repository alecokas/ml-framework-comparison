import torch
from torch.nn import Module
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


class FFNetwork(Module):
    def __init__(self, input_dims, hidden_dim, dropout_ratio, num_classes):
        super(FFNetwork, self).__init__()
        self.flat_image_dims = np.prod(input_dims)
        self.fc_1 = torch.nn.Linear(self.flat_image_dims, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.fc_2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, self.flat_image_dims)
        return self.fc_2(self.dropout(F.relu(self.fc_1(x))))


def get_dataloaders(batch_size, shuffle):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    training_set = datasets.MNIST('../data', train=True, transform=transform, download=True)
    test_set = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


def train_epoch(model, train_loader, optimiser):
    loss = torch.nn.CrossEntropyLoss()
    acc_loss = 0.0

    model.train()
    for batch_x, batch_y in train_loader:
        optimiser.zero_grad()
        pred_batch = model(batch_x)
        train_loss = loss(pred_batch, batch_y)
        train_loss.backward()
        optimiser.step()
        acc_loss += train_loss.item()
    return acc_loss / len(train_loader)


def accuracy(model, test_loader, batch_size):
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred_batch = model(batch_x)
            pred_batch = pred_batch.argmax(dim=1, keepdim=True)
            correct += pred_batch.eq(batch_y.view_as(pred_batch)).sum().item()

    return correct / (len(test_loader) * batch_size)


def main(num_epochs, batch_size, shuffle):
    train_loader, test_loader = get_dataloaders(batch_size, shuffle)

    model = FFNetwork(
        input_dims=(28, 28),
        hidden_dim=128,
        dropout_ratio=0.2,
        num_classes=10
    )
    optimiser = Adam(model.parameters())

    for epoch_idx in range(num_epochs):
        loss = train_epoch(model, train_loader, optimiser)
        print(f'Epoch {epoch_idx + 1} loss: {loss}')
    # Accuracy
    train_acc = accuracy(model, train_loader, batch_size)
    test_acc = accuracy(model, test_loader, batch_size)
    print(f'train_acc: {train_acc}  -  test_acc: {test_acc}')


if __name__ == '__main__':
    num_epochs = 5
    batch_size = 64
    shuffle = True
    main(num_epochs, batch_size, shuffle)
