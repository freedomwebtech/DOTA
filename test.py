import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 OBB model
model = YOLO('best.pt')             # Load trained YOLOv8 model (OBB-capable)
names = model.names                 # Get class names from the model

# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")  # Print coordinates when mouse moves

cv2.namedWindow("RGB")              # Create a named OpenCV window
cv2.setMouseCallback("RGB", RGB)    # Attach mouse callback to the window

# Open video file
cap = cv2.VideoCapture("test2.mp4") # Load input video
frame_count = 0                     # Initialize frame counter

while True:
    ret, frame = cap.read()         # Read a frame from video
    if not ret:
        break                       # Break if end of video

    frame_count += 1
    if frame_count % 3 != 0:
        continue                    # Skip every 2 out of 3 frames for performance

    frame = cv2.resize(frame, (1020, 500))  # Resize frame to fixed width and height

    # Run OBB tracking model
    results = model.track(frame, persist=True)  # Run YOLOv8 tracking with persistent IDs

    obbs = results[0].obb  # Get Oriented Bounding Boxes from result

    # If OBBs and polygon format exist
    if obbs is not None and obbs.xyxyxyxy is not None:
        for polygon, cls_id, conf in zip(obbs.xyxyxyxy, obbs.cls, obbs.conf):
            # Each polygon has 8 coordinates: [x1, y1, x2, y2, x3, y3, x4, y4]
            pts = polygon.view(4, 2).cpu().numpy().astype(np.int32)  # Reshape to 4x2 and convert to int

            # Draw green polygon
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw label at center
            cx, cy = np.mean(pts, axis=0).astype(int)  # Compute center of polygon
            label = f"{names[int(cls_id)]} {conf:.2f}"  # Create label with class name and confidence
            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)  # Draw label text in white

    # Show the frame
    cv2.imshow("RGB", frame)

    # Exit on ESC key
    if cv2.waitKey(0) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
