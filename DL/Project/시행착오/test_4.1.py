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
import queue
import keras
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit.runtime.runtime import Runtime, SessionInfo
import av
import tensorflow as tf
import plotly
import plotly.express as px

#학습된 모델 불러오기
MODEL_ROOT_DIR = "/Users/user/desktop/lab/project_2/models/"
MODEL_NAME = "Maded_model_3.h5"
MODEL_DIR = os.path.join(MODEL_ROOT_DIR,MODEL_NAME)
model = keras.models.load_model(MODEL_DIR)

#데이터 큐 생성 : 좌표 데이터를 저장할 공간
data_queue: queue.Queue = (
    queue.Queue()
) 

#33개 Landmark = pose landmark 를 가진 컬럼 생성
num_coords = 33 
landmarks = []
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
if 'landmark_df' in st.session_state:
    landmark_df = st.session_state.landmark_df
    if "Unnamed: 0" in landmark_df:
        df = df.drop(labels="Unnamed: 0",axis=1)
else:
    landmark_df = pd.DataFrame(columns=landmarks)
    
def main():
    st.header("1조 pose_tracking_project")

    pose_tracking_page = "데이터 입력"
    model_attempt_page = (
        "모델 적용"
    )

    app_mode = st.sidebar.radio(
        "모드를 선택하세요.",
        [
            pose_tracking_page,
            model_attempt_page,
        ],
    )
    st.subheader(app_mode)

    if app_mode == pose_tracking_page:
        app_pose_tracking()
    elif app_mode == model_attempt_page:
        app_model_attempt()

def app_pose_tracking():
    
    #drawing style 설정
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    #실시간 프로세싱 코드
    def process(image):
        image.flags.writeable = False 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # landmark 그리기
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
        try : 
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            row = pose_row

        except:
            pass

        return cv2.flip(image, 1), row

    # 서버
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # webrtc_streamer의 video_processor_factory 부분.
    class VideoProcessor:
        def recv(self, frame): 
            img = frame.to_ndarray(format="bgr24")
            img = process(img)[0]
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)[0]
        row = process(img)[1]
            
        #인식 못하는 row가 생긴다면 그 자리에 0을 넣어준다.
        if len(row) < len(landmarks):
            row = np.zeros(len(landmarks))
        df = pd.DataFrame(row).T
        df.columns=landmarks
        pred = model.predict(df)

        data_queue.put(pred)

        global landmark_df
        landmark_df = landmark_df.append(pd.Series(row,index=landmark_df.columns),ignore_index=True)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # 캠이 실행되는 부분
    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory = VideoProcessor,
        video_frame_callback = callback,
        async_processing=True,
    )
    
    # 사이드바에 체크박스 : 현재 좌표에서 모델을 적용시킨 값을 즉석으로 보여주고, 데이터프레임에 어떤 값이 들어가는지 가시적으로 보여줌
    if st.sidebar.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder_1 = st.empty()
            labels_placeholder_2 = st.empty()

            while True:
                try:
                    result = data_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None
                labels_placeholder_1.table(result)
                labels_placeholder_2.dataframe(landmark_df)
                
                # session_state 에 landmark_df 값을 저장
                st.session_state.landmark_df = landmark_df
        

def app_model_attempt():
    
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


#     # load the model
#     st.title("Load the Model")
#     stream = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')
#     if stream is not None:
#       myzipfile = zipfile.ZipFile(stream)
#       with tempfile.TemporaryDirectory() as tmp_dir:
#         myzipfile.extractall(tmp_dir)
#         root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
#         model_dir = os.path.join(tmp_dir, root_folder)
#         #st.info(f'trying to load model from tmp dir {model_dir}...')
#         model = tf.keras.models.load_model(model_dir)

#     # Load data from CSV
#     st.title("Upload your csv file.")
#     file_path = st.file_uploader("Upload a CSV file", type="csv")
#     if file_path is not None:

    global landmark_df
    if st.sidebar.button("측정값 확인"):
        
        a1 = st.empty()
        a1.dataframe(landmark_df)
                
    if st.sidebar.button("적용 시작"):        

        df = landmark_df
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


            
if __name__ == "__main__":
    main()