'''This is a demo python script which integrates the intel realsense D435i with object detection tasks'''

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

'''Define the global variables for visualization'''
CONFIDENCE_SCORE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
color = [255,0,0]
'''start the camera pipeline for realsense D435i'''

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def plot_results(detections, frame, depth_frame, color_intrinsic):
    boxes = detections[0].boxes.xyxy.tolist()
    classes = detections[0].boxes.cls.tolist()
    confidences = detections[0].boxes.conf.tolist()
    names = detections[0].names
    '''iterate through the results'''
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        name = names[int(cls)]
        '''getting the centre points in the bounding boxes in pixels. calculate the depth (z) get_distance() function. 
        This gives depth represented in camera coordinate system and 
        the bounding box centre coordinates are in pixels not meters and also in camera coordinate system'''
        dx, dy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        z_depth = depth_frame.get_distance(dx, dy)
        '''Now, we want to get the actual X, Y and Z for the object as a 3D point. We can use deproject from pixel to point method from intel SDK'''
        x, y, z = rs.rs2_deproject_pixel_to_point(color_intrinsic, [dx, dy], z_depth)

        object_loc = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})"
        print(object_loc)
        text = f"{name} ({confidence:.2f})"
        '''create rectangle for bouding box visualization'''
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)
        '''put the text'''
        cv2.putText(frame, text, (int(x1), int(y1)), FONT, 0.9, color=[255,0,0], thickness=2)
        cv2.putText(frame, object_loc, (dx,dy), FONT, 0.9, color=color, thickness=2)


    cv2.imshow('output', frame)

'''load the pretrained yolo model on coco128 for testing'''
model = YOLO('yolov8s.pt')
try:
    while True:
        frames= pipeline.wait_for_frames()
        '''get the RGB and Depth frame'''
        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not rgb_frame or not depth_frame:
            continue

        '''get the camera intrinsics'''
        colored_intrinsic = rs.video_stream_profile(rgb_frame.profile).get_intrinsics()
        '''convert the frames into numpy array to use for object detection'''
        colored_image = np.asanyarray(rgb_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        '''pass on the images to the model'''
        detection = model(colored_image)
        results = plot_results(detection, colored_image, depth_frame, colored_intrinsic)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
finally:
    pipeline.stop()

    cv2.destroyAllWindows()







