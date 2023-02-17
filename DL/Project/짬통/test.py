import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import time
import zipfile
import tempfile
import os
import tensorflow as tf
import plotly
import plotly.express as px


# sample code 1 (model predicttion to csv)
st.title('Pose reader')
# Add a placeholder progress bar
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    #update the progress bar with each iteration
    latest_iteration.text(f'iteration{i+1}')
    bar.progress(i+1)
    time.sleep(0.01)
    
    
# load the model

st.title("Load the Model")
stream = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')
if stream is not None:
  myzipfile = zipfile.ZipFile(stream)
  with tempfile.TemporaryDirectory() as tmp_dir:
    myzipfile.extractall(tmp_dir)
    root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
    model_dir = os.path.join(tmp_dir, root_folder)
    #st.info(f'trying to load model from tmp dir {model_dir}...')
    model = tf.keras.models.load_model(model_dir)
    
# Load data from CSV
st.title("Upload your csv file.")
file_path = st.file_uploader("Upload a CSV file", type="csv")
if file_path is not None:
    df = pd.read_csv(file_path)
    df = df.drop(labels="Unnamed: 0",axis=1)
    df2 = [df]

    # Make predictions
    pred = model.predict(df2)    
    results = []
    for i in pred:
        results.append(i.argmax()) # onehot decoding
    output = round(np.mean(results),4)    
    map_dict = {0: "left", 1: "middle", 2: "right"}
    output_2 = output - 1
    if output_2 > 0:
        bias="right"
    else:
        bias="left"
    st.title(f"Your Pose value is {output}. 0<=Value<=2, 0=left, 1=middle, 2=right. The value close to 1 is good.")
    st.title(f"Your Pose is {map_dict[round(output)]}, but {round(output_2,4)} biased to {bias}.")
    
    df["Predictions"] = results
    pred_name = []
    for i in results:
        pred_name.append(map_dict[i])
    df["Predictions_name"] = pred_name
    
    label = df["Predictions_name"].value_counts()             ## bar 차트 그리기
    st.dataframe( df.head() )
    st.bar_chart(label)
    #plotly pie차트
    df["Counts"] = 1
    df2 = df[["Predictions_name","Counts"]]
    data = df.groupby("Predictions_name").sum()
    fig1 = px.pie(data, values='Counts', names= data.index, title='자세 유형 분포도')
    st.plotly_chart(fig1)

    # Save predictions to new CSV file
    save_path = st.text_input("If you want to save the file, Please write the file path down.:")
    if st.button("Save"):
        df.to_csv(f"{save_path}.csv", index=False)
        st.success("File saved successfully!")

            
