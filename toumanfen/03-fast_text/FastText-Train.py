# -*- coding: utf-8 -*-
"""
@Name:FastText-Train
@author: itcast
todo: FastText-Train
@Time: 2024/10/28 16:12
"""

## (一)模型训练：版本一
# # 1.导入依赖包
# import fasttext
#
# # 指定训练集和测试集数据
# train_data_path = './data/data/train_fast.txt'
# test_data_path = './data/data/test_fast.txt'
#
# # 2.开启模型训练
# model = fasttext.train_supervised(input=train_data_path, wordNgrams=2)
# print('词的数量', len(model.words))
# print('标签值', model.labels)
#
# # 3.开启模型测试
# result = model.result(test_data_path)
# # 输出测试结果
# print(result)

# （二）FastText优化：自动化参数搜索（模型）
# 1.导入依赖包
import fasttext

# 2.数据及路径
train_data_path = './data/data/train_fast.txt'
test_data_path = './data/data/test_fast.txt'

# 3.模型训练
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2, autotuneValidationFile=dev_data_path,
                                       autotuneDuration=100, verbose=3)
result = model.test(test_data_path)
# print(result)

# 4.模型保存
model.save_model('./data/data/model/fasttext_model.bin')
