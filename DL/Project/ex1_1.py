import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import chardet
import pandas as pd
import streamlit as st
import queue
from streamlit.runtime.runtime import Runtime, SessionInfo
import keras
import os

#from aiortc.contrib.media import MediaRecorder

#데이터 큐 생성 : 좌표 데이터를 저장할 공간
data_queue: queue.Queue = (
    queue.Queue()
) 
#학습된 모델 불러오기
MODEL_ROOT_DIR = "/Users/user/desktop/lab/project_2/models/"
MODEL_NAME = "Maded_model_3.h5"
MODEL_DIR = os.path.join(MODEL_ROOT_DIR,MODEL_NAME)
model = keras.models.load_model(MODEL_DIR)

#drawing style 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#33개 Landmark = pose landmark 를 가진 컬럼 생성
num_coords = 33 
landmarks = []
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
landmark_df = pd.DataFrame(columns=landmarks)

#실시간 프로세싱 코드
def process(image):
    image.flags.writeable = False 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw the holistic annotations on the image.
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
        #인식 못하는 row가 생긴다면
        if len(row) < len(landmarks):
            row = np.zeros(len(landmarks))
        df = pd.DataFrame(row).T
        df.columns=landmarks
        pred = model.predict(df)
        # pred_value = pred.argmax()
        # map_dict = {0: "left", 1: "middle", 2: "right"}

    except:
        pass

    return cv2.flip(image, 1), pred, row


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame): 
        img = frame.to_ndarray(format="bgr24")
        img = process(img)[0]
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = process(img)[0]
    try :
        pred = process(img)[1]
        row = process(img)[2]
    except:
        pass
    data_queue.put(pred)
    
    global landmark_df
    landmark_df = landmark_df.append(pd.Series(row,index=landmark_df.columns),ignore_index=True)
    #print(data_queue.get(timeout=1.0))
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# def recorder_factory():
#     return MediaRecorder("record.wav")
# def recorder_factory() -> MediaRecorder:
#         return MediaRecorder(
#             "input.flv", format="flv")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory = VideoProcessor,
    video_frame_callback = callback,
    async_processing=True,
 #   in_recorder_factory=recorder_factory
)
if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder_1 = st.empty()
        labels_placeholder_2 = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            try:
                result = data_queue.get(timeout=1.0)
            except queue.Empty:
                result = None
            labels_placeholder_1.table(result)
            labels_placeholder_2.dataframe(landmark_df)
            

