"""
# Fasttext模型训练
# 1.导入依赖包
# 2.模型训练
# 3.模型测试
"""

# 1.导入依赖包
import fasttext

# 指定训练集和测试集数据
train_data_path = './data/data/train_fast.txt'
test_data_path = './data/data/test_fast.txt'

# 2.开启模型训练
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2)
print('词的数量', len(model.words))
print('标签值', model.labels)

# 3.开启模型测试
result = model.test(test_data_path)
# 输出测试结果
print(result)
