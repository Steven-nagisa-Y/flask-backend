import os.path
import threading
import uuid
from threading import Thread

from flask import *
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"
DOWNLOAD_DIR = "./downloads"

ALLOW_FILE = {'jpg', 'jpeg', 'png'}
HOST = "http://localhost:5050"

FUNC_NAMES = {
    "targetExtra": "目标提取",
    "targetDetect": "目标检测",
    "transDetect": "变换检测",
    "transDetectNew": "变换检测新图片",
    "terrainClassify": "地物分类",
}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOW_FILE


class targetExtra(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        # 这里处理目标提取图片逻辑
        # TODO
        lock.release()


class targetDetect(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        # 这里处理目标检测图片逻辑
        # TODO
        lock.release()


class transDetect(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.new_filename = filename.replace("transDetect", "transDetectNew")

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        # 这里处理变换检测图片逻辑
        # TODO
        lock.release()


class terrainClassify(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        # 这里处理地物分类图片逻辑
        # TODO
        lock.release()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/func/<name>', methods=['POST', 'GET'])
def handleFunc(name: str) -> object:
    """
    处理图片的总函数，可以改成静态路由
    :param name: 处理对象名称
    :return: JSON
    """
    func_name = name
    random_id = str(uuid.uuid4())[:8]
    if func_name not in FUNC_NAMES.keys():
        return {"errMsg": "请求路径出错", "status": 1}

    # 处理文件上传逻辑
    if request.method == 'POST':
        if 'file' not in request.files:
            return {"errMsg": "没有图片上传", "status": 1}
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return {"errMsg": "图片上传错误", "status": 1}
        # 有两个图片上传则需要保持id一致
        # 只在变换检测中，二次请求中携带id
        if 'id' in request.form:
            random_id = request.form.get('id')
        file_path = f"{func_name}-{random_id}-{secure_filename(file.filename)}"
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
        file.save(src_path)
        print("Uploaded file:", src_path)
        # 在这里可以处理图片了
        # 图片位置是：src_path 这个变量
        # 图片处理好之后放在 /downloads文件夹下，文件名不变为 file_path
        func_map = {
            "targetExtra": targetExtra,
            "targetDetect": targetDetect,
            "transDetect": transDetect,
            "terrainClassify": terrainClassify,
        }
        func = func_map[func_name](src_path)
        func.start()
        return {"errMsg": "", "status": 0,
                "data": {
                    "upload": f"{HOST}/uploads/{file_path}",
                    "download": f"{HOST}/downloads/{file_path}",
                    "id": random_id
                }}


@app.route("/uploads/<path>", methods=['GET'])
def get_upload(path: str) -> object:
    if request.method != 'GET':
        return {"errMsg": "请求方法错误", "status": 1}
    if not path:
        return {"errMsg": "无效请求文件", "status": 1}
    img_data = open(f"{app.config['UPLOAD_FOLDER']}/{path}", "rb").read()
    if not img_data:
        return {"errMsg": "无效请求文件", "status": 1}
    resp = make_response(img_data)
    resp.headers['Content-Type'] = 'image/png'
    return resp


@app.route("/downloads/<path>", methods=['GET'])
def get_download(path: str) -> object:
    if request.method != 'GET':
        return {"errMsg": "请求方法错误", "status": 1}
    if not path:
        return {"errMsg": "无效请求文件", "status": 1}
    img_data = open(f"{DOWNLOAD_DIR}/{path}", "rb").read()
    if not img_data:
        return {"errMsg": "无效请求文件", "status": 1}
    resp = make_response(img_data)
    resp.headers['Content-Type'] = 'image/png'
    return resp


if __name__ == '__main__':
    paths = ['uploads', 'downloads']
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
    app.run(host='0.0.0.0', port=5050, debug=True)
