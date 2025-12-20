import torchvision
from torchvision import transforms
from torchvision.datasets import STL10


def get_mnist_sets():
    image_resolution = (1, 28, 28)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
    return train_set, test_set, image_resolution


def get_cifar10_sets():
    image_resolution = (1, 32, 32)

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    return train_set, test_set, image_resolution

def get_stl10_sets():
    image_resolution = (3, 96, 96)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    train_dataset = STL10(
        root="./data",
        split="train",
        download=True,
        transform=transform_train
    )

    test_dataset = STL10(
        root="./data",
        split="test",
        download=True,
        transform=transform_test
    )

    return train_dataset, test_dataset, image_resolution