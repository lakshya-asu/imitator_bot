import cv2
import mediapipe as mp
import math
from pymycobot.mycobot import MyCobot

# Initialize MediaPipe Holistic.
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Initialize MyCobot.
mc = MyCobot("COMXX", 115200)

# Open Webcam.
cap = cv2.VideoCapture(0)

def calculate_angle(landmark1, landmark2, landmark3):
    # Calculate the angle between three landmarks.
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y

    radians = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    angle = abs(math.degrees(radians))

    return angle

while True:
    ret, frame = cap.read()

    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find body landmarks.
    results = holistic.process(rgb)

    # Draw body landmarks.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Calculate the joint angles for the right arm.
        shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

        shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)
        elbow_wrist_angle = calculate_angle(elbow, wrist, shoulder)

        # Send the joint angles to the myCobot 280.
        sensitivity = 1  # Adjust this value as needed.
        mc.send_angles([0, 50, shoulder_elbow_angle / sensitivity, elbow_wrist_angle / sensitivity, 0, 0], 50)

    # Display the image.
    cv2.imshow('MediaPipe Holistic', frame)

    # Exit if ESC key is pressed.
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Close the webcam and destroy all windows.
cap.release()
cv2.destroyAllWindows()
