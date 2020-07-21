import pandas as pd
import numpy as np
import streamlit as st
from os import listdir
from os.path import isfile, join
from PIL import Image
import testing
from keras.models import load_model
from keras import backend as K
import cv2

showpred = 0
try:
	model_path = 'finalModel.h5'
	model_weights_path = 'finalModel_weights.h5'
except: 
	print("Need to train model")
    
test_path = 'test/'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
st.sidebar.title("About")

st.sidebar.info(
    "The application identifies the pneumonia in the picture. It was built using a Convolution Neural Network (CNN).")

onlyfiles = [f for f in listdir("test/") if isfile(join("test/", f))]



st.sidebar.title("Predict New Images")
imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)



if st.sidebar.button('Predict Pneumonia'):
    showpred = 1
    prediction = testing.predict((model),"test/" + imageselect)


st.title('Pneumonia Identification')
st.write("Pick an image from the left. You'll be able to view the image.")
st.write("When you're ready, submit a prediction on the left.")

st.write("")
image = Image.open("test/" + imageselect)
st.image(image, caption="Let's predict the pneumonia!", use_column_width=True)

if showpred == 1:
    if prediction == 0:
        st.write("Yes")
    if prediction == 1:
        st.write("No")

        
