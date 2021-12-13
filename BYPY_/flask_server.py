# flask_server.py
# https://seokhyun2.tistory.com/43

# 모델을 서빙하는 소스코드
# 요청이 들어올 때마다 결과를 출력해서 반환하도록 구현
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request

from model import CNN


model = CNN()
model.load_state_dict(torch.load('mnist_model.pt'), strict=False)
model.eval()

normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

app = Flask(__name__)
@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    _, result = model.forward(normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
    return str(result.item())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2431, threaded=False)


# 서버에 요청하는 소스코드
# flask_test.py

import json
import requests
import numpy as np
from PIL import Image

image = Image.open('test_image.jpg')
pixels = np.array(image)

headers = {'Content-Type':'application/json'}
address = "http://127.0.0.1:2431/inference"
data = {'images':pixels.tolist()}

result = requests.post(address, data=json.dumps(data), headers=headers)

print(str(result.content, encoding='utf-8'))


# 딥러닝 모델 서빙과 병렬처리
# https://seokhyun2.tistory.com/44?category=932160
# 프로세스로 진행해야 함

