"""
## 数据集分析  random_forest.py
# 1.导入依赖包
# 2.读取数据集
# 3.构建语料库
# 4.获取停用词
# 5.计算tfidf特征stopwords.txt
# 6.划分数据集
# 7.实例化模型
# 8.模型训练
# 9.模型评估
"""

# 1.导入依赖包
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from icecream import ic
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# 2.读取数据集
# 指定数据集的位置
TRAIN_CORPUS = './data/data/train_new.csv'
STOP_WORDS = './data/data/stopwords.txt'
WORDS_COLUMN = 'words'

content = pd.read_csv(TRAIN_CORPUS)

# 3.构建语料库
corpus = content[WORDS_COLUMN].values

# 4.获取停用词
stop_words = open(STOP_WORDS).read().split()

# 5.计算tfidf特征stopwords.txt
tfidf = TfidfVectorizer(stop_words=stop_words)
text_vectors = tfidf.fit_transform(corpus)
# print(tfidf.vocabulary_)
# print(text_vectors)
# 目标值
targets = content['label']

# 6.划分数据集
x_train, x_test, y_train, y_test = train_test_split(text_vectors, targets, test_size=0.2, random_state=0)

# 7.实例化模型
model = RandomForestClassifier()  # verbose=3


# 8.模型训练
model.fit(x_train, y_train)

# 9.模型评估
accuracy = accuracy_score(model.predict(x_test), y_test)
ic(accuracy)
