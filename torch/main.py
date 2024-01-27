import requests
import gzip
import shutil
import os
import numpy as np
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

currentDir = os.path.dirname(os.path.abspath(__file__))

def downloadFile(url, destination, outputFile):
    response = requests.get(url)
    path = os.path.join(currentDir, "../dataset/")
    
    #CREATE FILE AND DUMP DOWNLOAD DATA
    with open(os.path.join(path, destination), 'wb') as f:
        f.write(response.content)
    
    #UNZIP'
    with gzip.open(os.path.join(path, destination), 'rb') as f_in:
        with open(os.path.join(path, outputFile), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(os.path.join(path, destination))

downloadFile('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz', 'train-images.idx3-ubyte')
downloadFile('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz', 'train-labels.idx1-ubyte')
downloadFile('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-images.idx3-ubyte')
downloadFile('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 't10k-labels.idx1-ubyte')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6)
        self.conv2 = nn.Conv2d(32, 32, stride=2, kernel_size=6)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

with open(os.path.join(currentDir, "../dataset/train-images.idx3-ubyte"), 'rb') as f:
    magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
    trainImages = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows * num_cols)

with open(os.path.join(currentDir, "../dataset/train-labels.idx1-ubyte"), 'rb') as f:
    magic, num_labels = struct.unpack('>II', f.read(8))
    trainLabels = np.fromfile(f, dtype=np.uint8)

with open(os.path.join(currentDir, "../dataset/t10k-images.idx3-ubyte"), 'rb') as f:
    magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
    testImages = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows * num_cols)

with open(os.path.join(currentDir, "../dataset/t10k-labels.idx1-ubyte"), 'rb') as f:
    magic, num_labels = struct.unpack('>II', f.read(8))
    testLabels = np.fromfile(f, dtype=np.uint8)


#PREPROCESSING THE DATA
trainImages = trainImages / 255.0
testImages = testImages / 255.0

trainImages = torch.tensor(trainImages, dtype=torch.float32)
trainLabels = torch.tensor(trainLabels, dtype=torch.long)
testImages = torch.tensor(testImages, dtype=torch.float32)
testLabels = torch.tensor(testLabels, dtype=torch.long)

trainImages = trainImages.reshape(-1, 1, 28, 28)
testImages = testImages.reshape(-1, 1, 28, 28)

trainDataset = TensorDataset(trainImages, trainLabels)
testDataset = TensorDataset(testImages, testLabels)

trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=64)

#INSTANTIATING THE MODEL
model = CNN().to("mps")
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#TRAINING
os.system("clear")
print("Starting Traing....")

startTime = time.time()
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in trainLoader:
        images, labels = images.to("mps"), labels.to("mps")
        optimizer.zero_grad()
        outputs = model(images)
        loss = lossFunction(outputs, labels) # Calculates loss
        loss.backward() #Backpropagates through network and computes gradients
        optimizer.step() #Updates Model parameters based on computed gradients
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Completed training in ", int(time.time() - startTime), "s")

#EVAL
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testLoader:
        images, labels = images.to("mps"), labels.to("mps")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")