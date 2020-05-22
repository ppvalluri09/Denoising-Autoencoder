from torchvision import datasets, transforms
import torch

def get_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST("", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    return train_loader
