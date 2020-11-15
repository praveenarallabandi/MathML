import torch
import torch.nn as nn
import torchvision
import torchvision.transforms
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
num_epochs = 5
num_classes = 10
batch_size = 50
learning_rate = 0.001

# 1.Loading and Normalizing MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)
test_dataset= torchvision.datasets.MNIST(root='.data/',
                                             train=False, 
                                             transform=transform,
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=500, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=500, 
                                           shuffle=False)
                                           
data_iter = iter(train_loader)
''' mean = data[0].mean()
std = data[0].std()
mean, std '''
images, label = data_iter.next()

print(label[0])

class MyNetBN(nn.Module):
    def __init__(self): 
        super(MyNetBN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 10)
        )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

cnn = MyNetBN()
print(cnn)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters()) 

#PyTorch - Training the Model
train_loss = list()
val_loss = list()
lossPlotX = []
epochPlotY = []
for epoch in range(5):
    # define the loss value after the epoch
    total_train_loss = 0
    total_val_loss = 0

    loss = 0.0
    number_of_sub_epoch = 0
    
    # loop for every training batch (one epoch)
    for images, labels in train_loader:
        #create the output from the network
        out = cnn(images)
        # count the loss function
        loss = criterion(out, labels)
        # in pytorch you have assign the zero for gradien in any sub epoch
        optimizer.zero_grad()
        # count the backpropagation
        loss.backward()
        # learning
        optimizer.step()
        # add new value to the main loss
        loss += loss.item()
        number_of_sub_epoch += 1
    finalLoss = loss / number_of_sub_epoch
    lossPlotX.append(finalLoss.detach().numpy())
    epochPlotY.append(epoch)    
    print("Result - Epoch {}: Loss: {}".format(epoch, finalLoss))
    #PyTorch - Comparing the Results
correct = 0
total = 0
cnn.eval()
for images, labels in test_loader:
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the {} test images: {}% with PyTorch'.format( total, 100 * correct / total ))

print(lossPlotX)
print(epochPlotY)
plt.scatter(lossPlotX, epochPlotY)
""" plt.scatter(X_test.data.numpy(), Y_test.data.numpy(), c='yellow', alpha=0.5, label='test') """
plt.show()
