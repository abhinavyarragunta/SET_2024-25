import cv2
import logging
from fall_detection_system import FallDetectionSystem

def main():
    """
    Main function to run the Fall Detection System.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Path to the YOLO pose estimation model
    model_path = '../yolo11x-pose (1).pt'

    # Initialize the Fall Detection System
    fall_detection_system = FallDetectionSystem(model_path)

    # Initialize video capture from the default webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    logging.info("Fall Detection System is running. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Error: Could not read frame from webcam.")
            break

        # Process the frame for fall and injury detection
        processed_frame = fall_detection_system.process_frame(frame)

        # Display the processed frame in a window
        cv2.imshow('Fall Detection System', processed_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting Fall Detection System.")
            break

    # Release the video capture object and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()