import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Normalize

# Implement the sequential module for feature extraction
torch.manual_seed(50)
network1 = nn.sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernal_size=5),
    nn.relu(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=12, kernal_size=5),
    nn.relu(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.flattern(start_dim=1),
    nn.linear(in_features=12 * 4 * 4, out_features=120),
    nn.relu(),
    nn.linear(in_features=20, out_features=60),
    nn.relu(),
    nn.linear(in_features=60, out_features=10))
torch.manual_seed(50)
network2 = nn.sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernal_size=5),
    nn.relu(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.BatchNorm2d(6),
    nn.Conv2d(in_channels=6, out_channels=12, kernal_size=5),
    nn.relu(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.flattern(start_dim=1),
    nn.linear(in_features=12 * 4 * 4, out_features=120),
    nn.relu(),
    nn.BatchNorm1d(120),
    nn.linear(in_features=120, out_features=60),
    nn.relu(),
    nn.linear(in_features=60, out_features=10))

#Loading and Normalizing MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))

loader = DataLoader(train_dataset,
                    batch_size=500,
                    num_workers=1)
data = next(iter(loader))
mean = data[0].mean()
std = data[0].std()
mean, std
train_set_normal = torchvision.datasets.MNIST(root='./data/',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      mean, std)
                                              ])
                                              )

trainsets = {'not_normal': train_dataset, 'normal': train_set_normal}
networks = {'no_batch_norm': network1, 'batch_norm': network2}
params = OrderedDict(learning_rate=[0.01],
                     batch_size=[1000],
                     num_workers=[1],
                     device=['cuda'],
                     trainset=['normal'],
                     network=list(networks.keys())
                     )
m = RunManager()
for run in RunBuilder.get_runs(params):
    device = torch.device(run.device)
    network = networks[run.network].to(device)
    loader = DataLoader(
        trainsets[run.trainset], batch_size=run.batch_size, num_workers=run.num_workers)
    optimizer = optim.Adam(network.parameters(),
                           learning_rate=run.learning_rate)
m.begin_run(run, network, loader)
for epoch in range(5):
    m.begin_epoch()
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        pred = network(images)
        loss = F.cross_entropy(pred, labels)  # calculating the loss
        optimizer.zero_grad()  # zero gradients
        loss.backward()
        optimizer.step()
m.track_loss(loss, batch)
m.track_num_correct(pred, labels)
m.end_run()
m.save('results')
pd.DataFrame.from_dict(m.run_data).sort_values('accuracy', ascending=False)
