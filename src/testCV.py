import cv2
import cvzone
from ultralytics import YOLO

# Load the pre-trained YOLOv11 model for pose estimation
model = YOLO('../../../yolo11x-pose (1).pt')  # Replace with the actual pose estimation model

# Open a connection to the webcam (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # If frame capture was unsuccessful, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run pose estimation on the captured frame
    results = model(frame)

    # Ensure results contain detections
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        keypoints = results[0].keypoints.xy.int().cpu().tolist()  # Joints
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        # Check the keypoints to ensure they have enough data
        if len(keypoints[0]) > 0:  # Ensure there are keypoints available
            object1 = keypoints[0]
            nose_y = object1[0][1]  # Y-coordinate of the nose
            if len(object1) > 16:  # Check if there are enough keypoints for ankles
                left_ankle_y = object1[15][1]  # Y-coordinate of the left ankle
                right_ankle_y = object1[16][1]  # Y-coordinate of the right ankle
                avg_ankle_y = (left_ankle_y + right_ankle_y) / 2  # Average y-coordinate of ankles
            else:
                avg_ankle_y = None  # Handle case where there are not enough keypoints for ankles
                print("Ankles not detected in the keypoints.")
        else:
            nose_y = None  # Handle case where there are no keypoints

        # Print nose and ankle information
        print(f"Nose Y: {nose_y}, Average Ankle Y: {avg_ankle_y}")

        for box, keypoint_set, class_id in zip(boxes, keypoints, class_ids):
            # Ensure keypoints have the necessary indices
            print("Compartmentalize features")
            if len(keypoint_set) > 16:  # Check that we have at least 17 keypoints
                print("Keypoints detected")
                x1, y1, x2, y2 = box
                nose_y = keypoint_set[0][1]
                left_ankle_y = keypoint_set[15][1]
                right_ankle_y = keypoint_set[16][1]
                avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
                print(nose_y, avg_ankle_y)

                #nose_y = 150
                #avg_ankle_y = 150
                if abs(nose_y - avg_ankle_y) <= 200:
                    # Mark as fall
                    print("Fallen")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cvzone.putTextRect(frame, f'{class_id}', (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, "Fall", (int(x2/2), int(y2/2)), 1, 1)
                else:
                    # Mark as normal
                    print("Not fallen")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{class_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, "Normal", (int(x2/2), int(y2/2)), 1, 1)
        else:
            print("No boxes or keypoints detected.")
    else:
        print("No detections available.")

    # Visualize the results on the frame
    result_frame = results[0].plot()

    # Display the frame with the pose estimation overlay
    cv2.imshow('YOLOv11 Pose Estimation', result_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
