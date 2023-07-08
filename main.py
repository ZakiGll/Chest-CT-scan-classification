import gradio as gr
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os

IMG_HEIGHT = 150
IMG_WIDTH = 150

model = load_model(os.path.join('models', 'Chest_CT_scan_classification.h5'))

classes = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

def classify_image(inp):
    resize = tf.image.resize(inp, (IMG_HEIGHT, IMG_WIDTH))
    prob = model.predict(np.expand_dims(resize / 255, 0))
    msg = "Predicted class : "+ str(classes[np.argmax(prob)])+ " "+ str(np.max(prob) * 100)+ "%"
    return msg

gr.Interface(classify_image,gr.Image(shape=(224, 224)), "text").launch()