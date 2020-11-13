import torch
import torch.nn as nn
import torchvision
import torchvision.transforms
import numpy as np 
import matplotlib.pyplot as plt
num_epochs = 5
num_classes = 10
batch_size = 50
learning_rate = 0.001

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                             train=True, 
                                             transform=transforms,
                                             download=True)
test_dataset= torchvision.datasets.MNIST(root='.data/',
                                             train=False, 
                                             transform=transforms,
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=32, 
                                           shuffle=False)
                                           
data_iter = iter(train_loader)
images, label = data_iter.next()

print(label[0])

#PyTorch - Building the Model
class ConvNet(nn.Module):
    def __init__(self, num_of_class):
        super(ConvNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1,6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc_model = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(84, 10)

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc_model(x)
        x = self.classifier(x)
        return x
    #PyTorch - Visualizing the Model
cnn = ConvNet(10)
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
        lossPlotX.append(loss)
        epochPlotY.append(epoch)
    print("Result - Epoch {}: Loss: {}".format(epoch, loss / number_of_sub_epoch))
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

#print(lossPlotX)
#print(epochPlotY)
plt.plot(lossPlotX, epochPlotY, 'o', color='red')
plt.show()
""" fig = plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, num_epochs + 1), train_loss, label="Train loss")
plt.plot(np.arange(1, num_epochs + 1), val_loss, label="Validation loss")
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show() """
