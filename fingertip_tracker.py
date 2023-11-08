import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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

    # Process the image and find hand landmarks.
    results = hands.process(rgb)

    # Draw hand landmarks of each hand.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the tip of the index finger.
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            print(f'Index Finger Tip Coordinates: x={x}, y={y}')

    # Display the image.
    cv2.imshow('MediaPipe Hands', frame)

    # Exit if ESC key is pressed.
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Close the webcam and destroy all windows.
cap.release()
cv2.destroyAllWindows()
