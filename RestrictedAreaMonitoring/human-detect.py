import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, filedialog
import os

points = []

# Mouse callback function to capture the coordinates of the clicked points
def mouse_click(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONDOWN:  # If left button is clicked
        points.append((x, y))
        print(f"Point recorded: ({x}, {y})")
        # Draw a small circle where you clicked
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("YOLO Human Detection", frame)

def get_left_lane_roi(frame):
    cv2.imshow("YOLO Human Detection", frame)
    cv2.setMouseCallback("YOLO Human Detection", mouse_click, frame)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return np.array(points)

def get_roi_mask(frame, roi_points):
    mask = np.zeros(frame.shape[:-1], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_points)], 255)
    return mask

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo12n.pt")

# Create a Tkinter root window (but keep it hidden)
root = Tk()
root.withdraw()

# Open file dialog to select video file
input_video_name = filedialog.askopenfilename(
    initialdir=os.getcwd(),
    title="Select Video File",
    filetypes=(
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*")
    )
)

if not input_video_name:
    print("No file selected. Exiting...")
    exit()

# Start video capture (0 for webcam or provide the path to your video file)
cap = cv2.VideoCapture(input_video_name)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

ret, frame = cap.read()

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer to save the output video with the specified properties
output_video_name = f"out-{os.path.basename(input_video_name).split('.')[0]}.mp4"
out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# get restircted area ROI
restricted_area_roi = get_left_lane_roi(frame)
restricted_area_mask = get_roi_mask(frame, restricted_area_roi)
cv2.imshow("Restricted Area Mask", restricted_area_mask)
print("Review the restricted area mask. Close the window when ready to proceed...")
cv2.waitKey(0)  # Wait until any key is pressed
cv2.destroyWindow("Restricted Area Mask")  # Close the mask window

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

    # Loop through the detected objects
    for idx, bbox in enumerate(results[0].boxes.xyxy):  # Coordinates are in the form [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2 = bbox
        cls = results[0].boxes.cls[idx]
        conf = results[0].boxes.conf[idx]

        # Check if the detected class is 'person' (class 0 in COCO dataset)
        if int(cls) == 0:
            # Create a polygon for the person's bounding box
            person_box = np.array([
                [int(x1), int(y1)],
                [int(x2), int(y1)],
                [int(x2), int(y2)],
                [int(x1), int(y2)]
            ])
            
            # Check for any overlap between person box and ROI
            is_within = False
            # Check if any point of the person's box is within the ROI
            for point in person_box:
                if (point[1] < restricted_area_mask.shape[0] and point[1] >= 0 and 
                    point[0] < restricted_area_mask.shape[1] and point[0] >= 0):
                    if restricted_area_mask[point[1], point[0]] == 255:
                        is_within = True
                        break
            
            if is_within:
                # Draw a bounding box for humans (person class)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f'Person- Danger {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, ), 2)
                cv2.putText(frame, f'Person - Safe {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.polylines(frame, [restricted_area_roi], isClosed=True, color=(0, 0, 255), thickness=3)
    
    # Add "Restricted Area" label above the ROI
    # Calculate the center of the ROI for text placement
    roi_center = np.mean(restricted_area_roi, axis=0).astype(int)
    # Get text size for proper positioning
    text = "Restricted Area"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Position text above the ROI center
    text_position = (roi_center[0] - text_width//2, roi_center[1] - 20)
    # Add text with black outline for better visibility
    cv2.putText(frame, text, text_position, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, text_position, font, font_scale, (0, 0, 255), thickness)
    
    # Write the annotated frame to the output video
    out.write(frame)
    # Display the output frame
    cv2.imshow('YOLO Human Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
print("output video saved as ", output_video_name)
out.release()
cap.release()
cv2.destroyAllWindows()
