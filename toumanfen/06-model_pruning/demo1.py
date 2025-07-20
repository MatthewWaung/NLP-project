# 导入工具包
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# 创建一个LeNet网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1: 图像的输入通道(1是黑白图像), 6: 输出通道数量, 3 * 3: 卷积核的尺寸
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
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

# module = model.conv1
# print(list(module.named_parameters()))

# 打印一个特殊的属性张量
# print(list(module.named_buffers()))
# print('-----------------------------------------------------------')

# 执行剪枝操作
# 第一个参数: module, 代表要进行剪枝的特定模块, 之前我们已经指定了module=model.conv1, 说明要对第一个卷积层执行剪枝操作
# 第二个参数: name, 指定要对选中的模型快中的哪些参数执行剪枝, 如果指定了name="weight", 意味着对连接网络中的weight剪枝, 不对bias进行剪枝
# 第三个参数: amount, 指定要对模型中多大比例的参数执行剪枝, 介于0.0-1.0之间的float数值, 代表百分比
# prune.random_unstructured(module, name="weight", amount=0.3)


# print(list(module.named_parameters()))
# print(list(module.named_buffers()))
# print('--------------------------------------------------------------')

# print(module.weight)


# 注意: 第三个参数amount, 当为一个float值代表的是剪枝的百分比, 如果为整数值, 代表剪裁掉的绝对数量
# prune.l1_unstructured(module, name="bias", amount=3)

# print(list(module.named_parameters()))
# print('----------------------------------------------------------------')
# print(list(module.named_buffers()))
# print('----------------------------------------------------------------')
# print(module.bias)
# print('----------------------------------------------------------------')

# 首先将原始模型的状态字典打印出来
# print(model.state_dict().keys())
# print('-------------------------------------------------------------------')

# 直接执行剪枝操作
# prune.random_unstructured(module, name="weight", amount=0.3)
# prune.l1_unstructured(module, name="bias", amount=3)

# 然后将剪枝后的模型的状态字典打印出来
# print(model.state_dict().keys())

# 打印剪枝后的模型参数
# print(list(module.named_parameters()))
# print('---------------------------------------------------------------------')

# 打印剪枝后的模型mask buffers参数
# print(list(module.named_buffers()))
# print('---------------------------------------------------------------------')

# 打印剪枝后的模型weight属性值
# print(module.weight)
# print('----------------------------------------------------------------------')


# 执行模型剪枝的永久花操作
# prune.remove(module, 'weight')
# print('----------------------------------------------------------------------')

# 打印执行remove之后的模型参数
# print(list(module.named_parameters()))
# print('----------------------------------------------------------------------')

# 打印执行remove之后的mask buffers参数
# print(list(module.named_buffers()))
# print('----------------------------------------------------------------------')


# 打印初始模型的mask buffers张量字典名称
print(dict(model.named_buffers()).keys())
print('------------------------------------------------------------------------')

# 打印初始模型的所有状态字典
print(model.state_dict().keys())
print('------------------------------------------------------------------------')

# 对模型进行分模块的参数的剪枝
for name, module in model.named_modules():
    # 对模型中所有的卷积层执行l1_unstructured剪枝操作, 选取20%的参数进行剪枝
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.2)
    # 对模型中所有的全连接层执行ln_structured剪枝操作, 选取40%的参数进行剪枝
    elif isinstance(module, torch.nn.Linear):
        prune.ln_structured(module, name="weight", amount=0.4, n=2, dim=0)

# 打印多参数模块剪枝后的mask buffers张量字典名称
print(dict(model.named_buffers()).keys())
print('------------------------------------------------------------------------')

# 打印多参数模块剪枝后模型的所有状态字典名称
print(model.state_dict().keys())
