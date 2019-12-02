import numpy as np
import keras.models
from keras.models import model_from_json
from flask import Flask, jsonify     
import imageio
import tensorflow as tf

def init():
    # load the architecture of the neural net
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load the weight of each neurons
    loaded_model.load_weights('model.h5')
    print('loaded model from disk')

    # compile and evaluate loaded_model
    loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    graph =  tf.compat.v1.get_default_graph()
    
    return loaded_model, graph