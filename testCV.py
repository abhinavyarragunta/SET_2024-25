import cv2
import cvzone
import numpy as np
import math
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)


class PoseEstimator:
    """
    Estimates human poses using a YOLO model and returns detection results.

    Attributes:
        model (YOLO): The YOLO model for pose estimation.
    """

    def __init__(self, model_path):
        """
        Initializes the PoseEstimator with the given model path.

        Args:
            model_path (str): Path to the YOLO pose estimation model.
        """
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Performs pose detection on a given frame.

        Args:
            frame (numpy.ndarray): The image frame to process.

        Returns:
            tuple: Contains boxes, keypoints, keypoint_scores, class_ids,
                   confidences, and the result object.
        """
        results = self.model(frame)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.int().cpu().tolist()
            keypoint_scores = results[0].keypoints.conf.cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            return boxes, keypoints, keypoint_scores, class_ids, confidences, results[0]

        return None, None, None, None, None, None


class FallDetector:
    """
    Detects falls based on keypoint positions and body orientation.

    Attributes:
        fall_threshold_ratio (float): Ratio for nose-to-ankle height comparison.
        angle_threshold (float): Angle threshold for torso orientation in degrees.
        consecutive_frames (int): Number of consecutive frames to confirm a fall.
        fall_frames (int): Counter for consecutive fall frames.
    """

    def __init__(self, fall_threshold_ratio=0.6, angle_threshold=70, consecutive_frames=2):
        """
        Initializes the FallDetector with specified parameters.

        Args:
            fall_threshold_ratio (float): Ratio for height comparison.
            angle_threshold (float): Angle threshold in degrees.
            consecutive_frames (int): Frames required to confirm a fall.
        """
        self.fall_threshold_ratio = fall_threshold_ratio
        self.angle_threshold = angle_threshold
        self.consecutive_frames = consecutive_frames
        self.fall_frames = 0

    def is_fallen(self, keypoints, keypoint_scores):
        """
        Determines if a person has fallen based on keypoints.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.

        Returns:
            bool: True if a fall is detected, False otherwise.
        """
        try:
            # Required keypoints with minimum confidence
            required_keypoints = {
                0: 0.5,   # Nose
                5: 0.5,   # Left Shoulder
                6: 0.5,   # Right Shoulder
                11: 0.5,  # Left Hip
                12: 0.5,  # Right Hip
                15: 0.5,  # Left Ankle
                16: 0.5   # Right Ankle
            }

            # Check keypoint confidences
            for idx, min_conf in required_keypoints.items():
                if keypoint_scores[idx] < min_conf:
                    logging.info(f"Low confidence for keypoint {idx}")
                    return False

            # Keypoints positions
            nose_y = keypoints[0][1]
            left_ankle_y = keypoints[15][1]
            right_ankle_y = keypoints[16][1]
            avg_ankle_y = (left_ankle_y + right_ankle_y) / 2

            height = abs(nose_y - avg_ankle_y)
            threshold = self.fall_threshold_ratio * height

            # Check if nose is near ankle level
            nose_near_ankles = abs(nose_y - avg_ankle_y) < threshold

            # Torso angle calculation
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]

            shoulder_midpoint = np.mean([left_shoulder, right_shoulder], axis=0)
            hip_midpoint = np.mean([left_hip, right_hip], axis=0)

            delta_x = hip_midpoint[0] - shoulder_midpoint[0]
            delta_y = hip_midpoint[1] - shoulder_midpoint[1]
            angle = abs(math.degrees(math.atan2(delta_y, delta_x)))

            if angle > 90:
                angle = 180 - angle

            torso_near_horizontal = angle < self.angle_threshold

            # Logging for debugging
            logging.debug(f"Torso angle: {angle}")
            logging.debug(f"Nose near ankles: {nose_near_ankles}")
            logging.debug(f"Torso near horizontal: {torso_near_horizontal}")

            # Determine fall
            if nose_near_ankles or torso_near_horizontal:
                self.fall_frames += 1
            else:
                self.fall_frames = 0

            return self.fall_frames >= self.consecutive_frames

        except IndexError as e:
            logging.warning(f"Keypoints missing for fall detection: {e}")
            return False


class InjuryAnalyzer:
    """
    Analyzes potential injuries based on color detection and posture analysis.

    Attributes:
        red_intensity_threshold (int): Threshold for red intensity in injury detection.
        red_difference_threshold (int): Threshold for red difference in injury detection.
        curl_up_distance_threshold (int): Distance threshold for curled-up posture detection.
        limping_ankle_diff_threshold (int): Ankle Y-coordinate difference threshold for limping detection.
    """

    def __init__(self, red_intensity_threshold=130, red_difference_threshold=70,
                 curl_up_distance_threshold=80, limping_ankle_diff_threshold=20):
        """
        Initializes the InjuryAnalyzer with specified thresholds.

        Args:
            red_intensity_threshold (int): Red intensity threshold.
            red_difference_threshold (int): Red difference threshold.
            curl_up_distance_threshold (int): Distance threshold for curled-up detection.
            limping_ankle_diff_threshold (int): Ankle difference threshold for limping detection.
        """
        self.red_intensity_threshold = red_intensity_threshold
        self.red_difference_threshold = red_difference_threshold
        self.curl_up_distance_threshold = curl_up_distance_threshold
        self.limping_ankle_diff_threshold = limping_ankle_diff_threshold

    def detect_injury_color(self, frame, x1, y1, x2, y2):
        """
        Detects potential injuries based on red color intensity.

        Args:
            frame (numpy.ndarray): The image frame to analyze.
            x1, y1, x2, y2 (int): Coordinates of the bounding box.

        Returns:
            bool: True if an injury is detected, False otherwise.
        """
        injury_detected = False

        # Define areas for head and torso
        head_area = frame[y1:y1 + 50, x1:x2]
        torso_area = frame[y1 + 50:y1 + 150, x1:x2]

        avg_color_head = cv2.mean(head_area)[:3]
        avg_color_torso = cv2.mean(torso_area)[:3]

        # Check for high red intensity in head area
        if (avg_color_head[2] > self.red_intensity_threshold and
                (avg_color_head[2] - np.mean([avg_color_head[0], avg_color_head[1]])) > self.red_difference_threshold):
            injury_detected = True
            cv2.putText(frame, "Potential Head Injury", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Check for high red intensity in torso area
        if (avg_color_torso[2] > self.red_intensity_threshold and
                (avg_color_torso[2] - np.mean([avg_color_torso[0], avg_color_torso[1]])) > self.red_difference_threshold):
            injury_detected = True
            cv2.putText(frame, "Potential Torso Injury", (x1, y1 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return injury_detected

    def is_limping(self, keypoints, keypoint_scores):
        """
        Determines if a person is limping based on ankle positions.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.

        Returns:
            bool: True if limping is detected, False otherwise.
        """
        try:
            required_keypoints = {15: 0.5, 16: 0.5}  # Left and Right Ankles
            for idx, min_conf in required_keypoints.items():
                if keypoint_scores[idx] < min_conf:
                    logging.info(f"Low confidence for keypoint {idx}")
                    return False

            left_ankle_y = keypoints[15][1]
            right_ankle_y = keypoints[16][1]
            limping_detected = abs(left_ankle_y - right_ankle_y) > self.limping_ankle_diff_threshold

            logging.debug(f"Limping detected: {limping_detected}")
            return limping_detected

        except IndexError as e:
            logging.warning(f"Keypoints missing for limping detection: {e}")
            return False

    def is_curled_up(self, keypoints, keypoint_scores):
        """
        Determines if a person is in a curled-up position.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.

        Returns:
            bool: True if curled-up posture is detected, False otherwise.
        """
        try:
            required_keypoints = {
                7: 0.5,   # Left Elbow
                8: 0.5,   # Right Elbow
                13: 0.5,  # Left Knee
                14: 0.5,  # Right Knee
                11: 0.5,  # Left Hip
                12: 0.5   # Right Hip
            }
            for idx, min_conf in required_keypoints.items():
                if keypoint_scores[idx] < min_conf:
                    logging.info(f"Low confidence for keypoint {idx}")
                    return False

            left_elbow = np.array(keypoints[7])
            right_elbow = np.array(keypoints[8])
            left_knee = np.array(keypoints[13])
            right_knee = np.array(keypoints[14])
            hip_center = np.mean([keypoints[11], keypoints[12]], axis=0)

            # Adjusted thresholds for detection
            elbows_near_hips = (
                np.linalg.norm(left_elbow - hip_center) < self.curl_up_distance_threshold and
                np.linalg.norm(right_elbow - hip_center) < self.curl_up_distance_threshold
            )

            knees_near_hips = (
                np.linalg.norm(left_knee - hip_center) < self.curl_up_distance_threshold and
                np.linalg.norm(right_knee - hip_center) < self.curl_up_distance_threshold
            )

            curled_up_detected = elbows_near_hips and knees_near_hips

            logging.debug(f"Curled up detected: {curled_up_detected}")
            return curled_up_detected

        except IndexError as e:
            logging.warning(f"Keypoints missing for curled-up detection: {e}")
            return False


class DisplayManager:
    """
    Manages the display of statuses and overlays on the video frames.
    """

    @staticmethod
    def draw_status(frame, box, fall_status, class_id, injury_status):
        """
        Draws the detection status and bounding box on the frame.

        Args:
            frame (numpy.ndarray): The image frame to draw on.
            box (list): Coordinates of the bounding box.
            fall_status (str): The fall status ("Fallen" or "Normal").
            class_id (int): ID of the detected person.
            injury_status (str): The injury status ("Injured" or "Normal").
        """
        x1, y1, x2, y2 = box
        color = (0, 255, 0)  # Default color: green

        if fall_status == "Fallen" or injury_status == "Injured":
            color = (0, 0, 255)  # Alert color: red

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        status_text = f"{fall_status}, {injury_status}"
        cvzone.putTextRect(frame, f'ID: {class_id}', (x1, y1 - 45), scale=1, thickness=1)
        cvzone.putTextRect(frame, status_text, (x1, y1 - 25), scale=1, thickness=1)


class FallDetectionSystem:
    """
    Main system class that integrates pose estimation, fall detection, and injury analysis.
    """

    def __init__(self, model_path):
        """
        Initializes the FallDetectionSystem with the specified model path.

        Args:
            model_path (str): Path to the YOLO pose estimation model.
        """
        self.pose_estimator = PoseEstimator(model_path)
        self.fall_detector = FallDetector()
        self.injury_analyzer = InjuryAnalyzer()
        self.display_manager = DisplayManager()

    def process_frame(self, frame):
        """
        Processes a single frame for fall and injury detection.

        Args:
            frame (numpy.ndarray): The image frame to process.

        Returns:
            numpy.ndarray: The processed frame with overlays.
        """
        boxes, keypoints_list, keypoint_scores_list, class_ids, confidences, result = self.pose_estimator.detect(frame)

        if boxes is None:
            logging.info("No detections available.")
            return frame

        for box, keypoints, keypoint_scores, class_id in zip(boxes, keypoints_list, keypoint_scores_list, class_ids):
            if len(keypoints) > 16:
                # Fall Detection
                fall_detected = self.fall_detector.is_fallen(keypoints, keypoint_scores)
                fall_status = "Fallen" if fall_detected else "Normal"

                # Injury Detection
                injured = False

                # Classify as injured if fallen
                if fall_detected:
                    injured = True

                # Detect potential injuries based on color
                if self.injury_analyzer.detect_injury_color(frame, *box):
                    injured = True

                # Check for limping
                if self.injury_analyzer.is_limping(keypoints, keypoint_scores):
                    injured = True

                # Check if the person is curled up
                if self.injury_analyzer.is_curled_up(keypoints, keypoint_scores):
                    injured = True

                injury_status = "Injured" if injured else "Normal"

                # Display Results
                self.display_manager.draw_status(frame, box, fall_status, class_id, injury_status)
            else:
                logging.info("Insufficient keypoints for detailed analysis.")

        # Overlay the skeletons on the frame
        frame_with_skeletons = result.plot()

        return frame_with_skeletons


def main():
    """
    Main function to run the Fall Detection System.
    """
    # Update the path to your model
    model_path = '../src/yolo11x-pose (1).pt'
    fall_system = FallDetectionSystem(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            break

        # Process frame using FallDetectionSystem
        processed_frame = fall_system.process_frame(frame)

        # Display frame
        cv2.imshow('Fall Detection System', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


