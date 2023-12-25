import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
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
    areas = []  # List to store the areas of bounding boxes
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Extract bounding box for cars and calculate area
        for det in results.xyxy[0]:
            if det[5] == 2:  # Class ID for 'car' in COCO dataset
                xmin, ymin, xmax, ymax = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                area = (xmax - xmin) * (ymax - ymin)
                areas.append(area)
                print(f"Frame {frame_id}: Car detected with bounding box [{xmin}, {ymin}, {xmax}, {ymax}], Area: {area}")

        frame_id += 1

    cap.release()
    return areas

# Process a specific video file and get areas
areas = process_video('/home/rmarri/speed-estimation/vehicle-speed-estimate-DL/CitroenC4Picasso_101.mp4')

# Plot the areas of the bounding boxes
plt.plot(areas)
plt.xlabel('Detection Number')
plt.ylabel('Area of Bounding Box')
plt.title('Area of Detected Cars in Video')
#plt.show()
plt.savefig('output_plot.png')
