from google.colab.patches import cv2_imshow

import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

class TrafficLight:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.state = "RED"  
    def set_state(self, state):
        self.state = state
        print(f"Lane {self.lane_id} Traffic Light: {self.state}")

def detect_traffic(video_path, interval_seconds=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)  

    lane_1_light = TrafficLight(1)
    lane_2_light = TrafficLight(2)

    frame_count = 0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = model(frame)
            vehicle_count = sum(1 for r in results[0].boxes if r.cls in [2, 3, 5, 7]) 
            # Simple traffic logic: If Lane 1 has more vehicles, give it green light
            if vehicle_count > 5: 
                lane_1_light.set_state("GREEN")
                lane_2_light.set_state("RED")
            else:
                lane_1_light.set_state("RED")
                lane_2_light.set_state("GREEN")

            annotated_frame = results[0].plot()
            cv2_imshow(annotated_frame)
            time.sleep(2)  
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'your video path location'  
    detect_traffic(video_path, interval_seconds=2)
