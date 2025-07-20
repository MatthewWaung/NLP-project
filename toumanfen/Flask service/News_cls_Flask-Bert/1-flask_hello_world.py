# 1.导入依赖包
from flask import Flask

# 2.实例化Flask
app = Flask(__name__)


# 3.定义请求响应函数
@app.route('/', methods=['GET'])
def index():
    return 'Hello World'


# 4.启动服务，监听5000端口
if __name__ == '__main__':
    app.run()
