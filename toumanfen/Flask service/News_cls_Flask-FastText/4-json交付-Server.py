# 1.导入依赖包
from flask import Flask, Response, json
from flask import request

# 2.实例化Flask服务
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 解决中文乱码问题


# 3.定义请求响应函数-路由1
@app.route('/NewsCls_handle', methods=['POST'])
def NewsCls_handle():
    # 3.1 获取json格式的输入
    request_json = request.get_json()
    content = request_json['content']

    # 3.2 定义响应数据格式
    respose_data = {
        'Status': 'success',
        "content": content,
    }
    # 3.3 返回请求数据
    return Response(status=200, response=json.dumps(respose_data, sort_keys=False))


# 4.启动服务，监听5000端口
if __name__ == '__main__':
    app.run()
