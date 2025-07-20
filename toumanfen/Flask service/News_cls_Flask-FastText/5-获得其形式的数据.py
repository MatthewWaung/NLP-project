# 1.导入依赖包
from flask import Flask
from flask import request

# 2.实例化Flask
app = Flask(__name__)


# 3.定义请求响应函数-路由1
@app.route('/userinfo_submit', methods=['GET'])
def userinfo_submit():
    with open('others_submit.html', 'rb') as file:
        content = file.read()

    return content

# 3.定义请求响应函数-路由1
@app.route('/userinfo_handle', methods=['POST'])
def userinfo_handle():
    name = request.form.get('name')
    age = request.form.get('age')
    sex = request.form.get('sex')

    print('姓名:', name)
    print('年龄:', age)
    print('性别:', sex)

    return 'Received Email Data!'

# 启动服务，监听5000端口
if __name__ == '__main__':
    app.run()
