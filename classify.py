import os
import sys
import time
import math
import numpy as np
import torchp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as Data
from torch.utils.data import dataset
import torchvision
from PIL import Image
# Model configuration, 3 layer NN model, all linear layers
class NN(torch.nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.hidden1 = nn.Linear(32*32*3,1000)
        self.hidden2 = nn.Linear(1000,100)
        self.out = nn.Linear(100,10)
        
    def forward(self, x):
        x = x.view(-1,32*32*3)
        
        #x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return x
#===========================================================#
# Use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'    
# Load training and testing datasets    
train_set = torchvision.datasets.CIFAR10(
    root='./data.CIFAR10', 
    train=True, 
    download=True, 
    transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size=100, 
    shuffle=True, 
    num_workers=1)

test_set = torchvision.datasets.CIFAR10(
    root='./data.CIFAR10', 
    train=False, 
    download=True, 
    transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(
    test_set, 
    batch_size=100, 
    shuffle=False, 
    num_workers=1)

# setup model, optimizer and loss function, here we use SGD and Cross Entropy loss
model = NN()
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Trainging machine, include 1 epoch training process and evaluation process.
def train_machine(epoch):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    # Training set evaluation #
    train_loss = 0
    train_acc = 0
    for data, target in train_loader:
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        train_loss += criterion(output, target)
        pred = output.max(1,keepdim = True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    train_loss /= len(train_loader.dataset)
    
    # Testing set evaluation #
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        test_loss += criterion(output, target)
        pred = output.max(1,keepdim = True)[1]
        test_acc += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    test_loss /= len(test_loader.dataset)
    print(#"|  {0:>}/10  |   {1:>.4f}   |   {2:>.4f}   |   {3:>.4f}  |  {4:>.4f}   |"
          "|{0: >2}/10|{1: ^15.4f}|{2: ^15.4f}|{3: ^15.4f}|{4: ^15.4f}|"
          .format(epoch+1, train_loss.item(), train_acc/50000*100,
                           test_loss.item(),  test_acc /10000*100))

    
# Trainging multiple epoches, and saveing the trained model
def train():
    print("start training")
    print("|{0: >5}|{1: ^15}|{2: ^15}|{3: ^15}|{4: ^15}|"
          .format("Loop","Train Loss","Train Acc %","Test Loss","Test Acc %"))
    for epoch in range(10):
        train_machine(epoch)
    save()

# Model saving function
def save():
    model_out_path = "./model/model.ckpt"
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))        
        
# Single image testing function     
def test():
    print("Test single image")
    # classes data
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
    image_path = sys.argv[2]
    # load model 
    model = torch.load("./model/model.ckpt")
    model.eval()
    # print(image_path)
    image = Image.open(image_path)
    # make the input image to tensor
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    #print(x.shape)
    output = model(x)
    pred = output.max(1,keepdim = True)[1]
    #print(pred.item())
    result = classes[pred.item()]
    print("predition result: {}".format(result))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
