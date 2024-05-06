
import ultralytics
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import copy

'''
pass the video frames to the pre trained yolo keypoint model on COCO.

track the output of the keypoint detector.

extract the trajectories of the tracked pedestrians according to the ids.

'''

import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

threshold = 0.25
# Store the track history
track_history = defaultdict(list)
def parse_hypothesis(results):

        hypothesis_list = []

        for box_data in results.boxes:
            class_id = int(box_data.cls)
            class_name = results[0].names
            hypothesis = {
                "class_id": class_id,
                "class_name": class_name,
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list


def parse_boxes(results):

        boxes_list = []
        for box_data in results.boxes:
            # get boxes values
            box = box_data.xywh[0]
            x = float(box[0])                                 # center x
            y = float(box[1])                                 # center y
            w = float(box[2])
            h = float(box[3])
            center_pos = (x,y)
            # append msg
            boxes_list.append(center_pos)

        return boxes_list

def parse_keypoints(results):

        keypoints_list = []
        for points in results.keypoints:

            keypoint_array = []
            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= threshold:

                    point_list = []
                    keypoint_id = kp_id + 1
                    x = float(p[0])
                    y = float(p[1])
                    score = float(conf)

                    point_info = {'keypoint_id': kp_id + 1, 'x': x, 'y': y, 'score': score}
                    keypoint_array.append(point_info)

            keypoints_list.append(keypoint_array)

        return keypoints_list

# Open the video file
video_path = "/home/rakshit/Downloads/3_08_29_00(1).mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # print(frame.shape)

    if success:
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker='/home/rakshit/Desktop/tracker.yaml', save = True , conf =0.25, imgsz = (1440,2560), classes = list(range(1)))  # specify class for only consider person

        results = results[0].cpu()

         # Visualize the results on the frame
        annotated_frame = results.plot()

        if results.boxes:
                hypothesis = parse_hypothesis(results)
                boxes = parse_boxes(results)
                # https://github.com/ultralytics/ultralytics/issues/4315
                track_ids = results.boxes.id.int().cpu().tolist()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    # x, y  = boxes[0], boxes[1]
                    track = track_history[track_id]
                    track.append((box[0], box[1]))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)
                # print(track_history)
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(30, 5, 2), thickness=1)

        
        if results.keypoints:
                keypoints = parse_keypoints(results)

       
        # if results.boxes.id is not None:
        #     # Get the boxes and track IDs
        #     boxes = results[0].boxes.xywh.cpu()

        #     # https://github.com/ultralytics/ultralytics/issues/4315
        #     track_ids = results.boxes.id.int().cpu().tolist()

        #     # Visualize the results on the frame
        #     annotated_frame = results[0].plot()

        #     # Plot the tracks
        #     for box, track_id in zip(boxes, track_ids):
        #         x, y, w, h = box
        #         track = track_history[track_id]
        #         track.append((float(x), float(y)))  # x, y center point
        #         if len(track) > 30:  # retain 90 tracks for 90 frames
        #             track.pop(0)

        #         # Draw the tracking lines
        #         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #         cv2.polylines(annotated_frame, [points], isClosed=False, color=(30, 5, 2), thickness=1)

     
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



