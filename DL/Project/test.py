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

# st.title("Image Processing")
# st.write(
#     "Upload an image below and I will annotate the faces in the image using [MediaPipe](https://mediapipe.dev/)."
# )

# file = st.file_uploader("Upload a file")
# if file is None:
#     file = st.camera_input("Or take a picture")

# if file is not None:
#     # Display original image
#     st.write("You uploaded", file.name)
#     col1, col2 = st.columns(2)
#     col1.image(file)

#     # Annotate image
#     file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)

#     with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         # Display annotate image
#         if results.multi_face_landmarks:
#             annotated_image = image.copy()

#             for face_landmarks in results.multi_face_landmarks:
#                 # st.code(face_landmarks)

#                 mp.solutions.drawing_utils.draw_landmarks(
#                     image=annotated_image,
#                     landmark_list=face_landmarks,
#                     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
#                 )
#                 mp.solutions.drawing_utils.draw_landmarks(
#                     image=annotated_image,
#                     landmark_list=face_landmarks,
#                     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
#                 )

#             col2.image(annotated_image, channels="BGR")
#         else:
#             col2.write("No faces found")

# sample code 1 (model predicttion to csv)
st.title('Pose reader')
# Load pre-trained model
model = load_model("Maded_model.h5")
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
stream = st.file_uploader('Maded_model.h5.zip', type='zip')
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
        results.append(i.argmax())
    output = np.mean(results)    
    map_dict = {0: "middle", 1: "left", 2: "right"}
    st.title(f"Your Pose is in the {output}. It means You're in the {map_dict[round(output)]} position.")
    
    df["Predictions"] = results
    pred_name = []
    for i in results:
        pred_name.append(map_dict[i])
    df["Predictions_name"] = pred_name

    # Save predictions to new CSV file
    save_path = st.text_input("If you want to save the file, Please write the file path down.:")
    if st.button("Save"):
        df.to_csv(f"{save_path}.csv", index=False)
        st.success("File saved successfully!")

            
# sample code 2 (csv to barplot)

# @st.cache
# def load_data(file_path):
#     return pd.read_csv(file_path)

# file_path = st.file_uploader("Upload a CSV file", type="csv")
# if file_path is not None:
#     df = load_data(file_path)
#     st.write("Columns:", list(df.columns))
#     x = st.selectbox("Select a column for X-axis", df.columns)
#     y = st.selectbox("Select a column for Y-axis", df.columns)
#     fig = plt.figure()
#     plt.bar(df[x], df[y])
#     st.write(fig)
    