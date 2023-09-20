import cv2
import streamlit as st
import numpy as np
import pyttsx3

from PIL import Image

from My_src.detection_keypoint import DetectKeypoint
from My_src.classification_keypoint import KeypointClassification

import datetime
check_time = datetime.datetime.now()
from check_pose0918 import check_pose
cp = check_pose()


# 初始化
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')

# # 語速控制
# rate = engine.getProperty('rate')
# engine.setProperty('rate', rate-20)

# # 音量控制
# volume = engine.getProperty('volume')
# engine.setProperty('volume', volume-0.25)



# import time

# Streamlit設定
st.set_page_config(
    layout="wide",
    page_title="Real-time Webcam Stream",
)
st.title("Real-time Webcam Stream")

# 開始捕獲攝像頭畫面stream
video_capture = cv2.VideoCapture(0)  # 0表示預設攝像頭

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 設定寬度
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 設定高度
video_capture.set(cv2.CAP_PROP_FPS, 30)  # 設定幀速率

# 設定要減小的畫面大小
new_width = 828  # 新的寬度
new_height = 621  # 新的高度

# 初始化關鍵點檢測和分類模型
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification(
    './My_models/pose_classification_epoches=400.pt'
)

# Create layout
#canvMain = st.empty()
can1, can2, can3 = st.columns(3)

ph1 = can1.empty()
ph2 = can2.empty()
ph3 = can3.empty()

image = Image.open("./My_images/catcow.gif")
ph1.image(image, caption="標準姿勢", use_column_width=True)

# 創建一個用於顯示攝像頭畫面的Streamlit元素
#frame_placeholder = st.empty()
#frame_placeholder = st.container()
olddetect=""
newdetect=""
while True:
   
    ret, frame = video_capture.read()  # 讀取攝像頭畫面
    if not ret:
        break

    # 調整畫面大小
    frame = cv2.resize(frame, (new_width, new_height))
    dtchange=0
    # 進行關鍵點檢測
    image_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 將畫面轉換為RGB格式
    try:
        results = detection_keypoint(image_cv)  # 使用關鍵點檢測模型進行檢測
        if "keypoints" not in results._keys:  #檢查是否存在關鍵點
            # 如果未檢測到關鍵點，繼續顯示攝像頭畫面並跳過後面的處理
            ph2.image(frame, channels="RGB", use_column_width=False)
            continue

        # 獲取關鍵點信息
        results_keypoint = detection_keypoint.get_xy_keypoint(results)

        # 進行姿勢分類
        input_classification = results_keypoint[10:]  # 選取用於分類的部分
        results_classification = classification_keypoint(input_classification)  # 使用關鍵點分類模型進行分類
        
        # 可視化結果
        image_draw = results.plot(boxes=False)  # 根據檢測結果繪製圖像
        
        a = results_classification[1].item()
        
        ph3.markdown(results_classification, unsafe_allow_html = True)
        print(results_classification)
        print(a)
        ph2.image(image_draw, channels="RGB", use_column_width=True)
       
            
        
    except IndexError:
        ph2.image(image_cv, channels="RGB", use_column_width=True)
    
    
# 關閉攝像頭
video_capture.release()
