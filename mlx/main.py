import requests
import gzip
import shutil
import os
import numpy as np
import struct
import time
import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn

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

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = mx.reshape(x ,[-1, 1024])
        x = nn.relu(self.fc1(x))
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

trainImages = mx.array(trainImages, dtype=mx.float32)
trainLabels = mx.array(trainLabels, dtype=mx.int64)
testImages = mx.array(testImages, dtype=mx.float32)
testLabels = mx.array(testLabels, dtype=mx.int64)

trainImages = trainImages.reshape(-1, 1, 28, 28)
testImages = testImages.reshape(-1, 1, 28, 28)

#INSTANTIATING THE MODEL
model = CNN()
optimizer = optim.Adam(learning_rate=0.001)

#TRAINING
os.system("clear")
print("Starting Traing....")

#Need to fix the error: ValueError: [conv] Expect the input channels in the input and weight array to match but got shapes - input: (1,1,28,28) and weight: (32,6,6,1)
startTime = time.time()
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, image in enumerate(trainImages):
        output = model(image.reshape(1,1,28,28))
        loss = nn.losses.cross_entropy(output, trainLabels)
        grads = nn.value_and_grad(model, loss)

        optimizer.update(model, grads)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Completed training in ", int(time.time() - startTime), "s")

# #EVAL
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in testLoader:
#         images, labels = images.to("mps"), labels.to("mps")
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(f"Test Accuracy: {100 * correct / total:.2f}%")