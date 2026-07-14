from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=128, transform=None, data_root="../data"):
    """Return train/test DataLoaders for MNIST."""
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    train_data = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
