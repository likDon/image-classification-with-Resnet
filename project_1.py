import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCH = 200
train_data_size, valid_data_size = 50000, 10000
LR = 0.1
C1 = 31
Net_Num_Blocks = [3,4,6,3]
history = []

Data_Aug = 'cutmix' # 'cutmix', 'cutout','simple'
Optim_type = 'momentum' #'SGD', 'momentum', 'Adam'
Scheduler_type = 'MultiStep' #'None', 'CLR', 'MultiStep'

file_path = './res/%s_%s_%s_%d_%s'%(Data_Aug, Optim_type, Scheduler_type, C1, ''.join([str(i) for i in Net_Num_Blocks]))

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def load_data(path):
    norm, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm, std)])
    
    if Data_Aug == 'cutout':
        train_transform.transforms.append(Cutout(n_holes = 1, length = 16))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm, std)])

    train_set = torchvision.datasets.CIFAR10(root = path, train = True, download = True, transform = train_transform)
    test_set = torchvision.datasets.CIFAR10(root = path, train = False, download = True, transform = test_transform)

    global train_data_size, valid_data_size
    train_data_size, valid_data_size = len(train_set), len(test_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1, pin_memory = True)
    return train_loader, test_loader

print('Load data..')
path = 'data'
train_loader, test_loader = load_data(path)
print('Train data: %d, Test data: %d.' % (train_data_size, valid_data_size))

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        global C1
        super(ResNet, self).__init__()
        self.in_planes = 31

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        layers = []
        for i in range(len(num_blocks)):
            if i == 0:
                layers.append(self._make_layer(block, C1 * (2**i), num_blocks[i], stride=1))
            else:
                layers.append(self._make_layer(block, C1 * (2**i), num_blocks[i], stride=2))
        self.layers = nn.Sequential(*layers)
        # self.layer1 = self._make_layer(block, 64, num_blocks[1], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(C1*(2**(len(num_blocks)-1)), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    global Net_Num_Blocks
    return ResNet(BasicBlock, Net_Num_Blocks)

print('Build model..')
net = project1_model().to(device)
print(sum(param.numel() for param in net.parameters() if param.requires_grad) / 1e6)

def validation():
    loss, correct = 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            out = net(input)
            loss += criterion(out, label).item() * input.size(0)
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == label).sum().item()
    return loss / valid_data_size, correct / valid_data_size

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train():
    global history
    model_path = file_path + '.pt'
    Max_valid_acc = 0.85
    criterion = nn.CrossEntropyLoss()
    if Optim_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr = LR)
    elif Optim_type == 'momentum':
        optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum=0.9, weight_decay = 5e-4)
    elif Optim_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = LR, betas=(0.9,0.99), weight_decay = 5e-4) # weight_decay 防止过拟合

    # optimizer = torch.optim.RMSprop(net.parameters(), lr = LR, alpha = 0.9)

    if Scheduler_type == 'CLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    elif Scheduler_type == 'MultiStep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
    
    history = []

    last_time = time.time()

    for epoch in range(EPOCH):
        train_loss, train_acc = 0.0, 0.0
        net.train()
        for input, label in train_loader:
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            
            if Data_Aug == 'cutmix':
                lam = np.random.uniform(0, 1)
                rand_index = torch.randperm(input.size()[0]).to(device)
                target_a = label
                target_b = label[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                # compute output
                out = net(input)
                loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
            else:
                out = net(input)
                loss = criterion(out, label)

            loss.backward()
            optimizer.step()
            if Scheduler_type != 'None':
                lr_scheduler.step()

            train_loss += loss.item() * input.size(0)
            _, predicted = torch.max(out.data, 1)
            train_acc += (predicted == label).sum().item()
        train_loss /= train_data_size
        train_acc /= train_data_size
        valid_loss, valid_acc = validation()
        if valid_acc > Max_valid_acc:
            Max_valid_acc = valid_acc
            torch.save(net.state_dict(), model_path)
        history.append([train_loss, valid_loss, train_acc, valid_acc])
        print('Epoch: %d, train_loss: %.03f, valid_loss: %.03f, train_acc: %.03f, valid_acc: %.03f' % (epoch, train_loss, valid_loss, train_acc, valid_acc))
        print('time:', time.time()-last_time)
        last_time = time.time()
    print('Best validation accuracy is: %.03f' % Max_valid_acc)
    return Max_valid_acc



print('Start training..')
train()
print('Training finished.')

print('Saving model...')
# model_path = './res/%s_%s_%s.pt'%(Data_Aug, Optim_type, Scheduler_type)
# torch.save(net.state_dict(), model_path)

np.save(file_path+'.npy', history)
print('Model saved.')

def plot(history):
    x = range(0, EPOCH)

    plt.title('Train/Test Loss & Accuracy vs Epochs')
    plt.subplot(2, 1, 1)
    plt.plot(history[:, :2])
    plt.legend(['Train Loss', 'Test Loss'])
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(history[:, 2:])
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.savefig(file_path+'.jpg')
    plt.show()

print('Plotting loss curve..')
plot(np.array(history))
