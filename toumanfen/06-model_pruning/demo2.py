import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1: 图像的输入通道(1是黑白图像), 6: 输出通道, 3x3: 卷积核的尺寸
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 是经历卷积操作后的图片尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet().to(device=device)

# 实现模型的全局剪枝

# 首先打印初始化模型的状态参数字典
print(model.state_dict().keys())
print('------------------------------------------------------------')

# 构建剪枝参数集合, 决定哪些层, 哪些参数会参与剪枝中
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'))

# 调用prune模块中的全局剪枝函数global_unstructured执行剪枝操作, 此处针对整体模型中的20%的参数进行剪枝
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)

# 最后打印简志厚的模型的状态字典
print(model.state_dict().keys())

# 全局剪枝后, 不同的层会有不同比例的权重参数被剪掉, 利用代码打印出来查看
print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    ))

print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    ))

print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    ))

print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    ))

print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    ))

# 最后再次打印一下全局的比例, 进行验证
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0)
                     + torch.sum(model.conv2.weight == 0)
                     + torch.sum(model.fc1.weight == 0)
                     + torch.sum(model.fc2.weight == 0)
                     + torch.sum(model.fc3.weight == 0))
        / float(model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement())
    ))
