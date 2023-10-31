import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


CONFIDENCE_THRESHOLD = 0.7
# Initialize the YOLOv8 model
model = YOLO('yolov8s.pt')
import math

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8s.pt")
align_to = rs.stream.color
align = rs.align(align_to)
try:
    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        color_intrinsic = rs.video_stream_profile(color_frame.profile).get_intrinsics()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        # Run YOLOv8 inference on the color frame
        results = model(color_image)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        #id = results[0].boxes.id.tolist()
        confidences = results[0].boxes.conf.tolist()
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            confidence = conf
            detected_classes = cls
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            cv2.rectangle(color_image, (x1,y1), (x2, y2), color=(255, 0,0), thickness=2)
            x, y = ((x1 + x2) / 2), ((y1 + y2) / 2)
            zDepth = depth_frame.get_distance(int(x), int(y))
            '''projcet the pixel to 3d point'''
            dx, dy, dz = rs.rs2_deproject_pixel_to_point(color_intrinsic, [x,y], zDepth)
            new_coord = f'({dx:.3f}, {dy:.3f}, {dz:.3f})'
            cv2.putText(color_image, new_coord, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            '''calculate the eulidean distance'''
            # distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
            # print('distnace from camera to pixel:', distance)
            print(('depth from camera to pixel:', zDepth))
            # Visualize the YOLOv8 results on the color frame
            # annotated_frame = results[0].plot()

            # Display the color frame with YOLOv8 annotations
            cv2.imshow('window_name', color_image)
            # cv2.imshow('depth_window', depth_colormap)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

finally:

    pipeline.stop()

    cv2.destroyAllWindows()