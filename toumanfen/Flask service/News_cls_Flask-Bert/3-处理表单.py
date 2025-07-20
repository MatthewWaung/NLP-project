# 1. 导入依赖包
from flask import Flask
from flask import request

# 2.实例化Flask服务
app = Flask(__name__)


# 3. 定义请求响应函数-路由1
@app.route('/NewsCls_submit', methods=['GET'])
def NewsCls_submit():
    with open('NewsCls_submit.html', 'rb') as file:
        content = file.read()

    return content


# 3.定义请求响应函数-路由2
@app.route('/NewsCls_handle', methods=['POST'])
def NewsCls_handle():
    email_data = request.form.get('content')
    print(email_data)

    return 'Received News Data!'


# 4.启动服务，监听5000端口
if __name__ == '__main__':
    app.run()
