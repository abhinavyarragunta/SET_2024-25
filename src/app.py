import cv2

cap = cv2.VideoCapture(0)  # Check this index
if not cap.isOpened():
    print("Cannot open camera")
