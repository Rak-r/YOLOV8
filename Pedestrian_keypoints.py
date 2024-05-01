from pathlib import Path
import ultralytics
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

'''
pass the video frames to the pre trained yolo keypoint model.

track the output of the keypoint detector.

extract the trajectories of the tracked pedestrians according to the ids.

Apply kalman filter to smooth out the trajectories.

'''

CONFIDENCE_THRESHOLD = 0.8

def detection_tracking(video:str):

    video = cv2.VideoCapture(1)

    if not video:
        print("Error opening video stream or file")
    
    while video.isOpened():

        ret, frames = video.read()
        if ret:
            model = YOLO('yolov8m-pose.pt')
            keypoints = model(frames)[0]
            
            cv2.imshow("RealSense YOLOv8 Tracking", frames)

        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

def main():

    detection_tracking(video='/home/rakshit/Downloads/ Retiro_caril_7_04_57_00.mp4')


if __name__ == '__main__':
    main()

