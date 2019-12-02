# from flask import Flask, render_template, request

# import imageio
# import numpy as np
# from tensorflow import keras
# from keras_preprocessing.image import load_img, img_to_array
# from flask import Flask, jsonify     
# import re
# import sys
# import os
# from PIL import Image
# import base64
# from cv2 import cv2
# from keras_preprocessing.image import load_img, img_to_array
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from keras.models import model_from_json
# import json
# from flask import Flask, render_template, request, jsonify
# import base64
# from predict import predictCharacter

# sys.path.append(os.path.abspath('./model'))
# # from load import *

# # init flask app
# app = Flask(__name__)

# # global model, graph
# # model, graph = init()

# # def convertImage(imgData1):
# #     imgstr = re.search(r'base64.(.*)', imgData1).group(1)
# #     with open('output.png', 'wb') as output:
# #         output.write(imgstr.decode('base64'))

# # def convertImage(imgData1):
# #     try:
# #         # searchbox_result = re.match("^.*(?=(\())", searchbox.group()
# #         imgstr = re.search(b'base64,(.*)',imgData1).group(1)
# #     except:
# #         imgstr = None
# #     with open('output.png','wb') as output:
# #         output.write(base64.b64decode(imgstr))

# def convertImage(imgData1):
#     imgstr = re.search(b'base64,(.*)',imgData1).group(1)
#     print(imgstr)
#     with open('output.png','wb') as output:
#         output.write(base64.b64decode(imgstr))


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods = ['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         data = request.get_json()
#         imagebase64 = data['image']
#         imgbytes = base64.b64decode(imagebase64)
#         with open("temp.png","wb") as temp:
#             temp.write(imgbytes)
#         result, prob = predictCharacter('temp.png')
        
#         print('HAHAHAHA')
#         print(result, ' ', prob)
#         return jsonify({
#             'prediction': str(result),
#             'probability': str(prob),
#             'status': True
#         })
#     # imgData = request.data
#     # print('HAHHAHA')
#     # print(imgData)
#     # print('HAEHEHE')

#     # convertImage(imgData)
#     # imgArr = cv2.imread("output.png", 0)
#     # imgArr = cv2.resize(imgArr, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#     # print(imgArr)
#     # imgArr = np.expand_dims(imgArr, axis=0)
#     # # new_model = tf.keras.models.load_model('my-model.model')
#     # # prediction = new_model.predict(imgArr)
#     # # return np.array_str(prediction[0])
#     # with graph.as_default():
#     #     print('ZXCVB')
#     #     out = model.predict(imgArr)
#     #     response = np.array_str(np.argmax(out))
#     # return response
    
# # def predict():
# #     print('HAHAHHAHA')
# #     # imgData = request.form.to_dict(flat=False)
# #     imgData = request.get_data()
# #     print('HIHIHIHIHI')
# #     print(type(imgData))
# #     convertImage(imgData)
# #     print('HUHUHUHUHU')
# #     x = imageio.imread('out.png', mode = 'L')
# #     print('HOHOHOHOHO')
# #     x = np.invert(x)
# #     print('HEHEHEHEH')
# #     x = np.array(Image.fromarray(x).resize(28, 28))
# #     print('QWERTY')
# #     x = x.reshape(1, 28, 28, 1)
# #     print('ASDFG')
# #     with graph.as_default():
# #         print('ZXCVB')
# #         out = model.predict(x)
# #         response = np.array_str(np.argmax(out))
# #     return response


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='127.0.0.1', port=port, debug=True)


# Second Try
from flask import Flask, render_template, request, jsonify
import base64
from predict import predict_character


app = Flask(__name__)

#default route
@app.route('/')
def index():
    return render_template('index.html', data = {'status': False})


@app.route('/charrecognize', methods = ['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        imagebase64 = data['image']
        imgbytes = base64.b64decode(imagebase64)
        with open("temp.png","wb") as temp:
            temp.write(imgbytes)
        result, prob = predict_character('temp.png')
        print(result, ' ', prob)

        return jsonify({
            'prediction': str(result),
            'probability': str(prob),
            'status': True
        })

if __name__ == "__main__":
    app.run(debug=True)