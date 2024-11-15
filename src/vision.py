import cv2
import cvzone
import numpy as np
import math
import logging
from ultralytics import YOLO

# Configure logging
#logging.basicConfig(level=logging.INFO)


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
        self.model = YOLO('yolo11x-pose.pt')

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

    def __init__(self, fall_threshold_ratio=0.6, angle_threshold=70, consecutive_frames=1):
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
        self.torso_angles = []
        self.torso_positions = []
        self.history_length = 5  # Number of frames to keep in history

    def update_torso_history(self, angle, position):
        """
        Updates the torso history buffers.

        Args:
            angle (float): Torso angle.
            position (float): Torso Y-position.
        """
        self.torso_angles.append(angle)
        self.torso_positions.append(position)

        if len(self.torso_angles) > self.history_length:
            self.torso_angles.pop(0)
            self.torso_positions.pop(0)

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
                0: 0.1,   # Nose
                5: 0.1,   # Left Shoulder
                6: 0.1,   # Right Shoulder
                11: 0.1,  # Left Hip
                12: 0.1,  # Right Hip
                15: 0.1,  # Left Ankle
                16: 0.1   # Right Ankle
            }

            # Check keypoint confidences and existence
            for idx, min_conf in required_keypoints.items():
                if idx >= len(keypoint_scores) or keypoint_scores[idx] < min_conf:
                    return False

            # Keypoints positions
            nose_y = keypoints[0][1]

            # Use available ankle keypoints
            ankle_ys = []
            if keypoint_scores[15] >= required_keypoints[15]:
                ankle_ys.append(keypoints[15][1])
            if keypoint_scores[16] >= required_keypoints[16]:
                ankle_ys.append(keypoints[16][1])

            if not ankle_ys:
                return False

            avg_ankle_y = np.mean(ankle_ys)

            height = abs(nose_y - avg_ankle_y)
            threshold = self.fall_threshold_ratio * height

            # Check if nose is near ankle level
            nose_near_ankles = abs(nose_y - avg_ankle_y) < threshold

            # Torso angle calculation using available keypoints
            shoulders = []
            if keypoint_scores[5] >= required_keypoints[5]:
                shoulders.append(keypoints[5])
            if keypoint_scores[6] >= required_keypoints[6]:
                shoulders.append(keypoints[6])

            hips = []
            if keypoint_scores[11] >= required_keypoints[11]:
                hips.append(keypoints[11])
            if keypoint_scores[12] >= required_keypoints[12]:
                hips.append(keypoints[12])

            if len(shoulders) < 1 or len(hips) < 1:
                return False

            shoulder_midpoint = np.mean(shoulders, axis=0)
            hip_midpoint = np.mean(hips, axis=0)

            delta_x = hip_midpoint[0] - shoulder_midpoint[0]
            delta_y = hip_midpoint[1] - shoulder_midpoint[1]
            angle = abs(math.degrees(math.atan2(delta_y, delta_x)))

            if angle > 90:
                angle = 180 - angle

            torso_near_horizontal = angle < self.angle_threshold
            # Update torso history
            self.update_torso_history(angle, hip_midpoint[1])

            if len(self.torso_angles) < self.history_length:
                # Not enough data yet
                return False

            # Compute the change in angle and position
            angle_change = self.torso_angles[-1] - self.torso_angles[0]
            position_change = self.torso_positions[-1] - self.torso_positions[0]

            fall_detected = (nose_near_ankles or torso_near_horizontal) and angle_change > 20 and position_change > 30

            if fall_detected:
                self.fall_frames += 1
            else:
                self.fall_frames = 0

            return self.fall_frames >= self.consecutive_frames

        except Exception as e:
            #logging.warning(f"Error in fall detection: {e}")
            return False

    def is_sitting(self, keypoints, keypoint_scores, frame_height):
        """
        Determines if a person is sitting on the ground or against a wall.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.
            frame_height (int): Height of the video frame.

        Returns:
            bool: True if sitting is detected, False otherwise.
        """
        try:
            required_keypoints = {11: 0.1, 12: 0.1}  # Left and Right Hips
            hips = []
            for idx, min_conf in required_keypoints.items():
                if idx < len(keypoint_scores) and keypoint_scores[idx] >= min_conf:
                    hips.append(keypoints[idx][1])
                else:
                    #logging.info(f"Low confidence or missing keypoint {idx}")
                    pass

            if not hips:
                return False

            avg_hip_y = np.mean(hips)

            # If hips are near the bottom of the frame, person might be sitting
            sitting_threshold = frame_height * 0.8  # Adjust as needed

            sitting_detected = avg_hip_y > sitting_threshold

            #logging.debug(f"Sitting detected: {sitting_detected}")
            return sitting_detected

        except Exception as e:
            #logging.warning(f"Error in sitting detection: {e}")
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
        self.left_ankle_history = []
        self.right_ankle_history = []
        self.history_length = 5  # Number of frames to keep in history

    def update_ankle_history(self, left_ankle_y, right_ankle_y):
        """
        Updates the ankle history buffers.

        Args:
            left_ankle_y (float): Y-position of the left ankle.
            right_ankle_y (float): Y-position of the right ankle.
        """
        self.left_ankle_history.append(left_ankle_y)
        self.right_ankle_history.append(right_ankle_y)

        if len(self.left_ankle_history) > self.history_length:
            self.left_ankle_history.pop(0)
            self.right_ankle_history.pop(0)

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
            required_keypoints = {15: 0.1, 16: 0.1}  # Left and Right Ankles
            available_ankles = []
            for idx, min_conf in required_keypoints.items():
                if idx < len(keypoint_scores) and keypoint_scores[idx] >= min_conf:
                    available_ankles.append(keypoints[idx][1])
                else:
                    #logging.info(f"Low confidence or missing keypoint {idx}")
                    pass

            if len(available_ankles) < 2:
                #logging.info("Not enough ankle keypoints for limping detection")
                return False

            left_ankle_y = keypoints[15][1]
            right_ankle_y = keypoints[16][1]

            self.update_ankle_history(left_ankle_y, right_ankle_y)

            if len(self.left_ankle_history) < self.history_length:
                # Not enough data yet
                return False

            # Compute the differences in ankle positions
            left_diffs = np.diff(self.left_ankle_history)
            right_diffs = np.diff(self.right_ankle_history)

            # Calculate the standard deviation of the differences
            left_std = np.std(left_diffs)
            right_std = np.std(right_diffs)

            std_ratio = max(left_std, right_std) / (min(left_std, right_std) + 1e-5)

            limping_detected = std_ratio > 1.5

            #logging.debug(f"Limping detected: {limping_detected}")
            return limping_detected

        except Exception as e:
            #logging.warning(f"Error in limping detection: {e}")
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
                7: 0.1,   # Left Elbow
                8: 0.1,   # Right Elbow
                13: 0.1,  # Left Knee
                14: 0.1,  # Right Knee
                11: 0.1,  # Left Hip
                12: 0.1   # Right Hip
            }
            available_elbows = []
            available_knees = []
            hips = []

            for idx, min_conf in required_keypoints.items():
                if idx < len(keypoint_scores) and keypoint_scores[idx] >= min_conf:
                    if idx in [7, 8]:
                        available_elbows.append(np.array(keypoints[idx]))
                    elif idx in [13, 14]:
                        available_knees.append(np.array(keypoints[idx]))
                    elif idx in [11, 12]:
                        hips.append(np.array(keypoints[idx]))
                else:
                    #logging.info(f"Low confidence or missing keypoint {idx}")
                    pass

            if len(hips) < 1:
                #logging.info("Not enough hip keypoints for curled-up detection")
                return False

            hip_center = np.mean(hips, axis=0)

            # Adjusted thresholds for detection
            elbows_near_hips = all(
                np.linalg.norm(elbow - hip_center) < self.curl_up_distance_threshold
                for elbow in available_elbows
            )

            knees_near_hips = all(
                np.linalg.norm(knee - hip_center) < self.curl_up_distance_threshold
                for knee in available_knees
            )

            curled_up_detected = elbows_near_hips and knees_near_hips

            #logging.debug(f"Curled up detected: {curled_up_detected}")
            return curled_up_detected

        except Exception as e:
            #logging.warning(f"Error in curled-up detection: {e}")
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
            #logging.info("No detections available.")
            # Reset histories
            self.fall_detector.torso_angles.clear()
            self.fall_detector.torso_positions.clear()
            self.injury_analyzer.left_ankle_history.clear()
            self.injury_analyzer.right_ankle_history.clear()
            return frame

        frame_height, frame_width = frame.shape[:2]

        for box, keypoints, keypoint_scores, class_id in zip(boxes, keypoints_list, keypoint_scores_list, class_ids):
            if len(keypoints) > 16:
                # Fall Detection
                fall_detected = self.fall_detector.is_fallen(keypoints, keypoint_scores)
                sitting_detected = self.fall_detector.is_sitting(keypoints, keypoint_scores, frame_height)

                if fall_detected or sitting_detected:
                    fall_status = "Fallen"
                else:
                    fall_status = "Normal"

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
                #logging.info("Insufficient keypoints for detailed analysis.")
                pass

        # Overlay the skeletons on the frame
        frame_with_skeletons = result.plot()

        return frame_with_skeletons


def main():
    """
    Main function to run the Fall Detection System.
    """
    # Update the path to your model
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        #logging.error("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            #logging.error("Error: Could not read frame.")
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


