import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from filelock import FileLock
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray


def load_data(data_dir="./data"):
    # In CI, short-circuit to the fake-data smoke set (see `load_test_data`
    # below). The real CIFAR-10 download from cs.toronto.edu is slow enough
    # (~25 kB/s from the workspace network) to blow past the per-test timeout.
    # We can't just check an env var here: this function runs inside Ray tune
    # trials (i.e. on Ray workers) whose process env is not the driver's env,
    # so a shell-exported `CIFAR_USE_FAKE=1` in tests.sh never reaches this
    # code. Instead, tests.sh touches a marker file under ~/.parallel-
    # experiments-use-fake-data — ~ (`/home/ray`) is EFS-shared across all
    # nodes in an Anyscale workspace, so every Ray worker sees the same file.
    if (
        os.environ.get("CIFAR_USE_FAKE") == "1"
        or os.path.exists(os.path.expanduser("~/.parallel-experiments-use-fake-data"))
    ):
        return load_test_data()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


def load_test_data():
    # Load fake data for running a quick smoke-test.
    trainset = torchvision.datasets.FakeData(
        128, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.FakeData(
        16, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    return trainset, testset


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
