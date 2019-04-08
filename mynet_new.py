'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse

from utils import progress_bar

import numpy as np
import matplotlib
#matplotlib.use('AGG')
import matplotlib.pyplot as plt 

#parsing instrutions
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training') 
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# shuffle = wash card  num_workers = the number of threads
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
# in windows
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
# in windows
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv2 = nn.Conv2d( 64,  128, 3, padding=1)
        
        self.conv3 = nn.Conv2d( 128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d( 256, 512, 3, padding=1)


        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8 * 8 * 512, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        #v1.1
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn3(x)))
        
        x = self.conv3(x)
        x = F.relu(self.bn4(x))
        x = self.conv4(x)
        x = self.pool(F.relu(self.bn5(x)))

        #fully connect
        x = x.view(-1, 8 * 8 * 512)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


net = Net()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss() # CrossEntropyLoss
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 

def test(epoch):
    global best_acc
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    re_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            re_loss = test_loss/(batch_idx+1)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return re_loss, 100.*correct/total

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    re_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        re_loss = (train_loss/(batch_idx+1))

    return re_loss, 100.*correct/total

epoch_size = 200
train_acc = np.zeros(epoch_size)
test_acc = np.zeros(epoch_size)
train_loss = np.zeros(epoch_size)
test_loss = np.zeros(epoch_size)
lr_list = []
for i in range(50):
    lr_list.append(0.1)
for i in range(50):
    lr_list.append(0.05)
for i in range(40):
    lr_list.append(0.01)
for i in range(30):
    lr_list.append(0.005)  
for i in range(30):
    lr_list.append(0.001) 
for epoch in range(start_epoch, start_epoch+epoch_size):

    optimizer = optim.SGD(net.parameters(), lr=lr_list[epoch], momentum=0.9, weight_decay=5e-4) 
    train_loss[epoch], train_acc[epoch] = train(epoch)
    test_loss[epoch], test_acc[epoch] = test(epoch)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

lns1=ax1.plot(np.arange(epoch_size), train_loss,   color='blue', label="train_loss")
lns2=ax1.plot(np.arange(epoch_size), test_loss,    color='yellow', label="test_loss")
lns3=ax2.plot(np.arange(epoch_size), train_acc ,   color='red' , label="train_acc")
lns4=ax2.plot(np.arange(epoch_size), test_acc ,    color='green' , label="test_acc")

lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=7)
ax2.grid()
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')

plt.title("loss and accuracy")
plt.savefig('test.png')

plt.show()
