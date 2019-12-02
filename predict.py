# # from keras_preprocessing.image import load_img, img_to_array
# # import numpy as np
# # import tensorflow as tf
# # from keras.models import load_model
# # from keras.models import model_from_json
# # import json
# # import matplotlib.pyplot as plt

# # from load import *

# # # load model
# # # model_path = 'saved_models/model_num.h5'
# # # convnet = load_model(model_path)

# # global loaded_model, graph
# # loaded_model, graph = init()

# # nep_numbers = ['0', '1' ,'2' , '3', '4', '5', '6', '7', '8', '9']

# # def predictCharacter(image_file):
# #     global graph
# #     with graph.as_default():

# #         # imgArr = cv2.resize(imgArr, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
# #         # print(imgArr)
# #         # imgArr = np.expand_dims(imgArr, axis=0)

# #         image_loaded = load_img(image_file,target_size=(28,28))
# #         img_arr = (img_to_array(image_loaded)/255.0).reshape(1, 28, 28, 3)
# #         probabilities = loaded_model.predict(img_arr)
# #         pred = np.argmax(probabilities)
# #         print('pred: ', pred)
# #         return nep_numbers[pred], np.amax(probabilities)

# from keras_preprocessing.image import load_img, img_to_array
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from keras.models import model_from_json
# import json

# from load import *

# global loaded_model, graph
# loaded_model, graph = init()

# # load model
# # model_path = 'saved_models/model_num.h5'
# # convnet = load_model(model_path)
# # def load():
# #     json_file = open('model_num.json', 'r')
# #     loaded_model_json = json_file.read()
# #     json_file.close()
# #     loaded_model = model_from_json(loaded_model_json)

# #     #load the weight of each neurons
# #     loaded_model.load_weights('model_num.h5')
# #     print('loaded model from disk')

# #     # compile and evaluate loaded_model
# #     loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
# # graph =  tf.compat.v1.get_default_graph()

# nep_numbers = ['0', '1' ,'2' , '3', '4', '5', '6', '7', '8', '9']

# def predictCharacter(image_file):
#     global graph
#     with graph.as_default():

#         # imgArr = cv2.resize(imgArr, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#         # print(imgArr)
#         # imgArr = np.expand_dims(imgArr, axis=0)

#         image_loaded = load_img(image_file,target_size=(28,28))
#         img_arr = (img_to_array(image_loaded)/255.0).reshape(1, 28, 28, 3)
#         probabilities = loaded_model.predict(img_arr)
#         print('this is the prob: ', probabilities)
#         pred = np.argmax(probabilities)
#         return nep_numbers[pred], np.amax(probabilities)


# Second try
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json
import json


# load model
# model_path = 'saved_models/model_num.h5'
# convnet = load_model(model_path)

json_file = open('model_num.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load the weight of each neurons
loaded_model.load_weights('model_num.h5')
print('loaded model from disk')

# compile and evaluate loaded_model
loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
graph =  tf.compat.v1.get_default_graph()

nep_numbers = ['0', '1' ,'2' , '3', '4', '5', '6', '7', '8', '9']

def predict_character(image_file):
    global graph
    with graph.as_default():

        # imgArr = cv2.resize(imgArr, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        # print(imgArr)
        # imgArr = np.expand_dims(imgArr, axis=0)

        image_loaded = load_img(image_file,target_size=(28,28))
        img_arr = (img_to_array(image_loaded)/255.0).reshape(1, 28, 28, 3)
        probabilities = loaded_model.predict(img_arr)
        print('prob : ', probabilities)
        pred = np.argmax(probabilities)
        return nep_numbers[pred], np.amax(probabilities)