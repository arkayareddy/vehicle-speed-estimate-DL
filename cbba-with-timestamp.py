import cv2
import torch
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video
    frame_id = 0
    car_data = {}  # Dictionary to store data for each car

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps  # Current timestamp in seconds
        results = model(frame)

        for det in results.xyxy[0]:
            if det[5] == 2:  # Class ID for 'car' in COCO dataset
                # Assuming each detection is a new car, which might not be true in a real scenario
                car_id = len(car_data)  # Unique ID for each car
                xmin, ymin, xmax, ymax = map(int, det[:4])
                area = (xmax - xmin) * (ymax - ymin)

                if car_id not in car_data:
                    car_data[car_id] = {'t0': timestamp, 't_max': timestamp, 'max_area': area, 'exit_time': timestamp}
                else:
                    if area > car_data[car_id]['max_area']:
                        car_data[car_id]['t_max'] = timestamp
                        car_data[car_id]['max_area'] = area
                    car_data[car_id]['exit_time'] = timestamp

        frame_id += 1

    cap.release()
    return car_data

# Example usage
video_path = '/home/rmarri/speed-estimation/vehicle-speed-estimate-DL/CitroenC4Picasso_101.mp4'
car_data = process_video(video_path)

# Plotting
for car_id, data in car_data.items():
    plt.plot([data['t0'], data['t_max'], data['exit_time']], [0, data['max_area'], 0], label=f'Car {car_id}')

plt.xlabel('Time (seconds)')
plt.ylabel('Area of Bounding Box')
plt.title('Car Detections Over Time')
plt.savefig('output_plot1.png')