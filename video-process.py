import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the function to process the video
def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Extract bounding box for cars
        for det in results.xyxy[0]:
            if det[5] == 2:  # Class ID for 'car' in COCO dataset
                xmin, ymin, xmax, ymax = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                print(f"Frame {frame_id}: Car detected with bounding box [{xmin}, {ymin}, {xmax}, {ymax}]")

        frame_id += 1

    cap.release()

# Process a specific video file
process_video('CitroenC4Picasso_101.mp4')
