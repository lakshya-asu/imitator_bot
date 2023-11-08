from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pymycobot.mycobot import MyCobot
import cv2
import numpy as np
import math

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

# Initialize MyCobot
mc = MyCobot('COM7', 115200)  # Replace 'COM7' with the correct port for your setup

def calculate_angle(joint1, joint2, joint3):
    # Calculate the angle between three joints
    a = math.sqrt((joint2.x - joint1.x)**2 + (joint2.y - joint1.y)**2 + (joint2.z - joint1.z)**2)
    b = math.sqrt((joint2.x - joint3.x)**2 + (joint2.y - joint3.y)**2 + (joint2.z - joint3.z)**2)
    c = math.sqrt((joint3.x - joint1.x)**2 + (joint3.y - joint1.y)**2 + (joint3.z - joint1.z)**2)
    angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
    return math.degrees(angle)

while True:
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()
        frame = np.reshape(frame, (1080, 1920, 4))
        frame = cv2.resize(frame, (640, 480))

    # --- draw skeletons to color frame
    if kinect.has_new_body_frame(): 
        bodies = kinect.get_last_body_frame()

        if bodies is not None: 
            for i in range(0, kinect.max_body_count):
                body = bodies.bodies[i]
                if not body.is_tracked: 
                    continue 

                joints = body.joints
                # convert joint coordinates to color space 
                joint_points = kinect.body_joints_to_color_space(joints)
                # Get the joint positions
                shoulder = joints[PyKinectV2.JointType_ShoulderRight]
                elbow = joints[PyKinectV2.JointType_ElbowRight]
                wrist = joints[PyKinectV2.JointType_WristRight]

                # Calculate the angles
                shoulder_elbow_angle = calculate_angle(shoulder.Position, elbow.Position, wrist.Position)
                elbow_wrist_angle = calculate_angle(elbow.Position, wrist.Position, shoulder.Position)

                # Send the angles to MyCobot
                angles = [0, 0, shoulder_elbow_angle, elbow_wrist_angle, 0, 0]
                speed = 50  # Adjust the speed as needed
                mc.send_angles(angles, speed)

                # Display the angles
                print(f'Shoulder-Elbow Angle: {shoulder_elbow_angle}')
                print(f'Elbow-Wrist Angle: {elbow_wrist_angle}')

    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break
