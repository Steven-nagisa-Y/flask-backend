import http.server
import os.path
import socketserver
import threading
import uuid
from threading import Thread

from flask import *
from flask_cors import CORS
from paddlers.tasks.utils.visualize import visualize_detection
import numpy as np
import paddlers as pdrs
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app, resource=r"/*", supports_credentials=True)
app.config['UPLOAD_FOLDER'] = "./uploads"
DOWNLOAD_DIR = "./downloads"

ALLOW_FILE = {'jpg', 'jpeg', 'png'}
HOST = "http://localhost:5050"  # 如果要改端口号，改这里的5050

FUNC_NAMES = {
    "targetExtra": "目标提取",
    "targetDetect": "目标检测",
    "transDetect": "变换检测",
    "transDetectNew": "变换检测新图片",
    "terrainClassify": "地物分类",
}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOW_FILE


class TargetExtra(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.predictor = pdrs.deploy.Predictor('core/inference_model/oseg')

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        try:
            img = cv2.imread('./uploads/' + self.filename)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            # img_file参数指定输入图像路径
            result = self.predictor.predict(img_file=img)
            print(result['label_map'].shape)
            prob = result['label_map']
            result = ((prob > 0.5) * 255).astype('uint8')
            result = cv2.resize(result, (512, 512))  # 将输出图像大小改为640*480
            cv2.imwrite('./downloads/' + self.filename, result)  # 保存结果
        finally:
            lock.release()


class TargetDetect(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.predictor = pdrs.deploy.Predictor('core/inference_model/det')

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        try:
            img = cv2.imread('./uploads/' + self.filename)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            # img_file参数指定输入图像路径
            result = self.predictor.predict(img_file=img)
            vis = img
            if len(result) > 0:
                vis = visualize_detection(
                    np.array(vis), result,
                    color=np.asarray([[0, 255, 0]], dtype=np.uint8),
                    threshold=0.2, save_dir=None
                )
            # result = ((prob>0.5) * 255).astype('uint8')
            vis = cv2.resize(vis, (512, 512))  # 将输出图像大小改为640*480
            cv2.imwrite('./downloads/' + self.filename, vis)  # 保存结果
        finally:
            lock.release()


class TransDetect(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.new_filename = filename
        self.filename = filename.replace("transDetectNew", "transDetect")
        self.predictor = pdrs.deploy.Predictor('core/inference_model')

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        print("=============>", self.filename, self.new_filename)
        try:
            img = cv2.imread('./uploads/' + self.filename)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            img1 = cv2.imread('./uploads/' + self.new_filename)
            img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
            # img_file参数指定输入图像路径
            result = self.predictor.predict(img_file=(img, img1))
            print(result[0]['label_map'].shape)
            prob = result[0]['label_map']
            result = ((prob > 0.5) * 255).astype('uint8')
            result = cv2.resize(result, (512, 512))  # 将输出图像大小改为640*480
            cv2.imwrite('./downloads/' + self.filename, result)  # 保存结果
        finally:
            lock.release()


class TerrainClassify(Thread):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.predictor = pdrs.deploy.Predictor('core/inference_model/class')

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        try:
            img = cv2.imread('./uploads/' + self.filename)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

            # img_file参数指定输入图像路径
            result = self.predictor.predict(img_file=img)
            print(result['label_map'].shape)
            prob = result['label_map']
            lut = np.zeros((256, 3), dtype=np.uint8)
            lut[0] = [255, 0, 0]
            lut[1] = [30, 255, 142]
            lut[2] = [60, 0, 255]
            lut[3] = [255, 222, 0]
            lut[4] = [0, 0, 0]
            result = lut[prob]
            result = cv2.resize(result, (512, 512))  # 将输出图像大小改为640*480
            cv2.imwrite('./downloads/' + self.filename, result)  # 保存结果
        finally:
            lock.release()


class HTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="dist", **kwargs)


class HttpServer:
    def __init__(self, port=8000):
        super().__init__()
        self.port = port

    def run(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        print("HTTP Server running at PORT:", self.port)
        try:
            Handler = HTTPHandler
            Handler.extensions_map.update({
                ".js": "application/javascript",
            })
            httpd = socketserver.TCPServer(("", self.port), Handler)
            httpd.serve_forever()
        finally:
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
        # 图片处理好之后放在 /downloads文件夹下，文件名不变为 file_path
        func_map = {
            "targetExtra": TargetExtra,
            "targetDetect": TargetDetect,
            "transDetect": TransDetect,
            "terrainClassify": TerrainClassify,
        }
        if func_name == 'transDetect':
            return {"errMsg": "coutinue", "status": 0,
                    "data": {
                        "upload": f"{HOST}/uploads/{file_path}",
                        "download": f"{HOST}/downloads/{file_path}",
                        "id": random_id
                    }}
        if func_name == 'transDetectNew':
            func_name = 'transDetect'
        func = func_map[func_name](file_path)
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
    httpserver = HttpServer(port=8000)
    thread = Thread(target=httpserver.run)
    thread.start()
    app.run(host='0.0.0.0', port=5050, debug=True)
