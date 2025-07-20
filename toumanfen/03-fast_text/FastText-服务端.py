# -*- coding: utf-8 -*-
"""
@Name:Flask-demo
@author: itcast
todo: 程序的作用
@Time: 2024/10/28 17:19
"""
import fasttext
import jieba
# Flask服务化框架  Web框架     IP:port   封装predict.py


# 1.导入依赖包 Flask
from flask import Flask, request

# 2.实例化Flask对象
app = Flask(__name__)


# 3.请求响应函数
@app.route('/v1/fasttext', methods=['POST'])
def predict():
    # 加载自定义的停用词表
    jieba.load_userdict('./data/data/stopwords.txt')
    # 提供已经训练好的模型路径
    model_save_path = 'toutiao_fasttext_1699862718.bin'
    # 实例化fasttext对象, 并加载模型参数用于推断, 提供服务请求
    model = fasttext.load_model(model_save_path)
    print('FastText模型实例化完毕...')

    # 1.接受输入
    text = request.form['text']
    text_new = ' '.join(list(text))

    # 2.模型预测
    pred = model.predict(text_new)
    print(pred)
    result = pred[0][0]

    return result


# 4.启动Flask服务
if __name__ == '__main__':
    app.run()
