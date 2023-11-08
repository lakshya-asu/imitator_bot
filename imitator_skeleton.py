import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Holistic.
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Open Webcam.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find body landmarks.
    results = holistic.process(rgb)

    # Draw body landmarks of each hand.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Get the coordinates of the wrist and the tip of the index finger.
        wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
        index_finger_tip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX]

        # Calculate the angle.
        angle = math.degrees(math.atan2(index_finger_tip.y - wrist.y, index_finger_tip.x - wrist.x))
        print(f'Angle between wrist and index finger: {angle} degrees')

    # Display the image.
    cv2.imshow('MediaPipe Holistic', frame)

    # Exit if ESC key is pressed.
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Close the webcam and destroy all windows.
cap.release()
cv2.destroyAllWindows()
