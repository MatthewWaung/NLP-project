import re
import time

import fasttext
import jieba
from flask import Flask, request, Response, json

# 加载自定义的停用词表
jieba.load_userdict('data/stopwords.txt')
# 提供已经训练好的模型路径
model_save_path = 'model/toutiao_fasttext_1699865297.bin'
# 实例化fasttext对象, 并加载模型参数用于推断, 提供服务请求
model = fasttext.load_model(model_save_path)


# 创建Flask应用
app = Flask(__name__)


# 3.定义请求响应函数-路由1
@app.route('/NewsCls_submit', methods=['GET'])
def email_submit():
    with open('NewsCls_submit.html', 'rb') as file:
        content = file.read()

    return content


# 定义路由，接收POST请求并进行推理
@app.route('/NewsCls_handle', methods=["POST"])
def main_server():
    # 1.从POST请求中获取用户ID和文本数据
    request_json = request.get_json()
    content = request_json['content']

    # 2.调用推理函数获取预测结果
    t1 = time.time()
    # 1.数据预处理
    text_new = ' '.join(jieba.lcut(content))
    # 2.模型预测
    pred = model.predict(text_new)
    result = pred[0][0]

    # 结果输出re
    match = re.search(r'__label__(.+)', result)
    result = match.group(1)
    print(result)

    t2 = time.time()

    # 3.返回预测结果
    # return f'邮件类型：{prediction}'
    respose_data = {
        'Status': 'success',
        "Result": result,
        'Time': '{:.4f}s'.format(t2 - t1)
    }
    return Response(status=200, response=json.dumps(respose_data, sort_keys=False))


# 如果脚本作为主程序运行，则启动Flask应用
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5004)
