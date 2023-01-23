# 该版本的代码发布到了我的github上
# url: https://github.com/liulin0x3c/CNN-MNIST
import torch
from torch.utils.data import DataLoader
from torchvision import transforms  # 数据的原始处理
from torchvision import datasets
import torch.nn.functional as F  # 激活函数
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        # 第二个卷积层
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 池化层
        self.pooling = torch.nn.MaxPool2d(2)
        # 分类用的线性层
        self.fc = torch.nn.Linear(320, 10)

    # 下面就是计算的过程
    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)  # 这里面的0是x大小第1个参数，自动获取batch大小
        # 输入x经过一个卷积层，之后经历一个池化层，最后用relu做激活
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        # 为了给我们最后一个全连接的线性层用
        # 我们要把一个二维的图片（实际上这里已经是处理过的）20x4x4张量变成一维的
        x = x.view(batch_size, -1)  # flatten
        # 经过线性层，确定他是0~9每一个数的概率
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)  # lr为学习率


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:
            test()
