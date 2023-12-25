import cv2
import matplotlib.pyplot as plt
import gdown
import torch
from yolov5 import YOLOv5  # This is an example, adjust based on your YOLO version

# Function to download video from Google Drive
def download_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# Initialize YOLOv5
yolo_model = YOLOv5('yolov5n-seg.pt', device='cpu')  # or 'cuda' for GPU

# Function to process video frames and detect cars
def process_video(video_path):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_areas = []
    frame_times = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        print(f"yolo processing started")
        results = yolo_model.predict(frame)
        print(f"video processed with yolo")
        for det in results.xyxy[0]:  # Assuming results.xyxy contains detection data
            if det[-1] == 2:  # Assuming class '2' is for cars
                bbox = det[:4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Calculate area
                frame_areas.append(area)
                frame_times.append(frame_id)
                print(f"calculating area : {area}")

        frame_id += 1

    cap.release()
    print(f"Processed frame area: {frame_areas}")
    return frame_times, frame_areas

# Plotting function
def plot_area_change(frame_times, frame_areas):
    plt.plot(frame_times, frame_areas)
    plt.xlabel('Frame')
    plt.ylabel('Bounding Box Area')
    plt.title('Change in Car Bounding Box Area Over Time')
    plt.show()

# Main process
google_drive_file_id = '1mvTsLC06kq2hNwcEU2RizBkTRYwsCx_F'
video_path = 'CitroenC4Picasso_101.mp4'
download_from_google_drive(google_drive_file_id, video_path)

frame_times, frame_areas = process_video(video_path)
plot_area_change(frame_times, frame_areas)
