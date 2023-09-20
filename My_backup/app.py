import cv2
import numpy as np
import matplotlib.pyplot as plt

from My_src.detection_keypoint import DetectKeypoint
from My_src.classification_keypoint import KeypointClassification

# 创建检测关键点和分类关键点的实例
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification(
    './My_models/My_pose_classification.pt'
)

def pose_classification(frame):

    try:
        # detection keypoint
        results = detection_keypoint(frame)
        results_keypoint = detection_keypoint.get_xy_keypoint(results)

        # classification keypoint
        input_classification = results_keypoint[10:]
        results_classification = classification_keypoint(input_classification)

        # visualize result
        frame_draw = results.plot(boxes=False)
        x_min, y_min, x_max, y_max = results.boxes.xyxy[0].numpy()
        frame_draw = cv2.rectangle(
                        frame_draw, 
                        (int(x_min), int(y_min)),(int(x_max), int(y_max)), 
                        (0,0,255), 2
                    )
        (w, h), _ = cv2.getTextSize(
                        results_classification.upper(), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
        frame_draw = cv2.rectangle(
                        frame_draw, 
                        (int(x_min), int(y_min)-20),(int(x_min)+w, int(y_min)), 
                        (0,0,255), -1
                    )
        frame_draw = cv2.putText(frame_draw,
                        f'{results_classification.upper()}',
                        (int(x_min), int(y_min)-4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255),
                        thickness=2
                    )
        return frame_draw
    except IndexError:
        # 当出现 "IndexError: index 0 is out of bounds for axis 0 with size 0" 时，返回摄像机当前帧
        return frame

# 打开摄像头
cap = cv2.VideoCapture(0)

# 创建一个窗口
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while cap.isOpened():
    # 读取摄像头帧
    success, frame = cap.read()
    if success:
        frame_out = pose_classification(frame)

        # 在 'frame' 窗口中显示图像
        cv2.imshow('frame', frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()