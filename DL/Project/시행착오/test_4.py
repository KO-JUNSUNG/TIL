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
import csv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tensorflow as tf
import plotly
import plotly.express as px
import seaborn as sns


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic 
holistic = mp_holistic.Holistic(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

file = '/mnt/c/Users/KOJUNSUNG/Desktop/landmarks_coordinates.csv'

if os.path.isfile(file):
    pass
else:
    num_coords = 33
    landmarks = []
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]  
    with open(file, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)


def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
    #mp output = landmarks. 
    """
    Output
    Naming style may differ slightly across platforms/languages.

    POSE_LANDMARKS
    A list of pose landmarks. Each landmark consists of the following:

    x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
    z: Should be discarded as currently the model is not fully trained to predict depth, but this is something on the roadmap.
    visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
    """
    
    try:

        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        row = pose_row
        # landmarks saved.
        
        with open(file, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
    except:
        pass

    
    return cv2.flip(image, 1), results


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor: #for webrtc_ctx
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24") # frame -> array
        print(img.shape)
        img, results = process(img) # mediapipe holistic skeleton call , results = True or False
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")  # array -> video

# webrtc_ctx = webrtc_streamer( # cam code
#     key="WYH",
#     mode=WebRtcMode.SENDRECV,
#     rtc_configuration=RTC_CONFIGURATION,
#     media_stream_constraints={"video": True, "audio": False},
#     video_processor_factory=VideoProcessor,
#     async_processing=True,
# )

webrtc_ctx = webrtc_streamer(
key="WYH",
mode=WebRtcMode.SENDRECV,
rtc_configuration=RTC_CONFIGURATION,
media_stream_constraints={"video": True, "audio":False}, # 300 * 300 웹에 표현되는 크기
video_processor_factory = VideoProcessor,
async_processing=True,
)


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
    if "Unnamed: 0" in df:
        df = df.drop(labels="Unnamed: 0",axis=1)
    df2 = [df]
    df3 = df.copy()
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
    df4 = pd.DataFrame()
    for i in range(0,33):
        df5 = df3.iloc[:,[4*i, 4*i + 1]]
        df4 = pd.concat((df4,df5),ignore_index=True,axis=1)
    # df4 구성
    # 이제 df4 에다가 x*width, y*height 해서 좌표 구해주면 되겠다. 

    # Create a blank image to be used as the canvas for the heatmap
    canvas = np.zeros((300, 300), np.uint8)
    canvas.fill(255)
    # Loop through the landmark data
    for i in range(df4.shape[0]):
        for j in range(int(df4.shape[1]/2)):
            x, y = df4.iloc[i,[2*j, 2*j +1]]
            x = x * 300  #denormalize
            y = y * 300
            if x >= 0 and x <= 300 and y >= 0 and y <= 300:
                # Plot a circle on the canvas using OpenCV's circle function
                canvas[int(y),int(x)] += 1

    # Show the heatmap
    canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
    st.image(canvas, use_column_width=True)

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

            
