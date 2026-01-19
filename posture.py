import cv2 # camera and draws lines
import mediapipe as mp # framework for running AI
import numpy as np # vector calculator
import time # timestamps video frames

# lateral tilt threshold (experimentally determined)
TILT_THRESHOLD = 9 

# forward compression ratio threshold (experimentally determined)
RATIO_THRESHOLD = 0.45

CAMERA_ID = 1  # 1 = Mac, 0 = iPhone

# set up shortcuts
BaseOptions = mp.tasks.BaseOptions 
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# must point to correct file path
model_path = '/Users/alexsuhajda/Documents/Code/pose_landmarker.task'

# configure engine
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=lambda result, output_image, timestamp_ms: save_result(result)
)

# holds AIs latest response
latest_result = None

def save_result(result):
    global latest_result
    latest_result = result

def calculate_angle(a, b):
    # a = Head, b = Shoulders
    a = np.array(a)
    b = np.array(b)
    neck_vector = a - b # find vector from shoulder to head
    vertical_vector = np.array([0, -1]) 
    unit_neck = neck_vector / np.linalg.norm(neck_vector)
    unit_vertical = vertical_vector / np.linalg.norm(vertical_vector)
    radians = np.arccos(np.dot(unit_neck, unit_vertical))
    return np.degrees(radians)


cap = cv2.VideoCapture(CAMERA_ID)

with PoseLandmarker.create_from_options(options) as landmarker:
    print("Starting Posture Checker (Ratio + Angle). Press 'q' to quit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]
            
            # 1. get keypoints (eyes & shoulders)
            left_eye = [landmarks[2].x, landmarks[2].y]
            right_eye = [landmarks[5].x, landmarks[5].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]
            right_shoulder = [landmarks[12].x, landmarks[12].y]

            # 2. calculate centers & distances
            head_center = [(left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2]
            shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]

            # convert to pixels
            h, w, _ = image.shape
            head_px = np.array([head_center[0] * w, head_center[1] * h])
            shoulder_px = np.array([shoulder_center[0] * w, shoulder_center[1] * h])
            l_sh_px = np.array([left_shoulder[0] * w, left_shoulder[1] * h])
            r_sh_px = np.array([right_shoulder[0] * w, right_shoulder[1] * h])

            
            # a. lateral tilt (angle)
            neck_angle = calculate_angle(head_px, shoulder_px)

            # b. forward compression (ratio)
            # 1. calculate vertical neck height (Y distance only)
            neck_height = abs(shoulder_px[1] - head_px[1])
            # 2. calculate shoulder width (Euclidean distance)
            shoulder_width = np.linalg.norm(l_sh_px - r_sh_px)
            # 3. the golden ratio
            posture_ratio = neck_height / shoulder_width

            
            # Check 1: Is head tilted sideways?
            is_tilted = neck_angle > TILT_THRESHOLD
            # Check 2: Is head dropped forward? (ratio too low)
            is_compressed = posture_ratio < RATIO_THRESHOLD

            if is_tilted or is_compressed:
                status = "BAD POSTURE"
                color = (0, 0, 255) 
                
                # detailed error message
                if is_tilted: status += " (TILT)"
                if is_compressed: status += " (SLOUCH)"
            else:
                status = "GOOD POSTURE"
                color = (0, 255, 0) 

            
            cv2.line(image, (int(shoulder_px[0]), int(shoulder_px[1])), 
                            (int(head_px[0]), int(head_px[1])), color, 4)
            
            # display the stats on screen for calibration
            cv2.putText(image, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # show the raw numbers so you can find your threshold
            stats = f"Tilt: {int(neck_angle)} deg | Ratio: {posture_ratio:.2f}"
            cv2.putText(image, stats, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Posture Checker', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()