import datetime
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Initialize the RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8s.pt")

# Initialize the DeepSort tracker
tracker = DeepSort(max_age=50)

# Dictionary to store object IDs and their positions
object_ids = {}

while True:
    start = datetime.datetime.now()

    # Wait for a frame from the RealSense camera
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        break

    # Convert RealSense frames to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asarray(color_frame.get_data())

    # Run YOLOv8 object tracking on the color frame
    detections = model(color_image)[0]
    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), GREEN, 2)
    # Initialize the list of bounding boxes and confidences

    # End time to compute the fps
    end = datetime.datetime.now()
    # Show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # Calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(color_image, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # Show the frame to the screen
    cv2.imshow("RealSense YOLOv8 Tracking", color_image)

    if cv2.waitKey(1) == ord("q"):
        break

pipeline.stop()
cv2.destroyAllWindows()