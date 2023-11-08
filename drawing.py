import cv2
import mediapipe as mp
import tkinter as tk
from pymycobot.mycobot import MyCobot
from PIL import Image, ImageTk, ImageDraw
from threading import Thread

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Initialize MyCobot.
#mc = MyCobot("COMXX", 115200)

# Open Webcam.
cap = cv2.VideoCapture(0)

# Create a blank image for the drawing.
drawing = Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(drawing)

def start_drawing():
    global drawing
    while True:
        ret, frame = cap.read()

        # Check if frame is not None.
        if frame is None:
            print("Failed to read frame from webcam.")
            continue

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

                # Draw on the blank image.
                draw.point((x, y), fill='black')

                # Send the coordinates to the myCobot 280.
                #mc.send_coords([x, y, 0, 0, 0, 0], 50)

        # Display the image.
        cv2.imshow('MediaPipe Hands', frame)

        # Exit if ESC key is pressed.
        if cv2.waitKey(5) & 0xFF == 27:
            break

def stop_drawing():
    # Close the webcam and destroy all windows.
    cap.release()
    cv2.destroyAllWindows()

# Create the GUI.
root = tk.Tk()

# Create the start button.
start_button = tk.Button(root, text='Start Drawing', command=lambda: Thread(target=start_drawing).start())
start_button.pack()

# Create the stop button.
stop_button = tk.Button(root, text='Stop Drawing', command=stop_drawing)
stop_button.pack()

# Run the GUI.
root.mainloop()
