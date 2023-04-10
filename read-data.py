import cv2
import os
import re
import mediapipe as mp
import pandas as pd

# Initialize mediapipe lib
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "FALLBACK"
no_of_frames = 200

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    # Draw the line to connect the landmarks
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
  
    # Draw the landmarks on the frame
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return frame


# Set the path of all the videos
video_folder_fallback = '/home/yuu/Documents/PBL5-demo/Data/Fall_backwards'
video_folder_fallforward = '/home/yuu/Documents/PBL5-demo/Data/Fall_forward'
video_folder_fallleft = '/home/yuu/Documents/PBL5-demo/Data/Fall_left'
video_folder_fallright = '/home/yuu/Documents/PBL5-demo/Data/Fall_right'
video_folder_fallsitting = '/home/yuu/Documents/PBL5-demo/Data/Fall_sitting'
video_folder_walk = '/home/yuu/Documents/PBL5-demo/Data/Walk'

# Read video from the folder and save the landmarks to a csv file
def read_video(video_folder):
    for video in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video)
        cap = cv2.VideoCapture(video_path)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, frame = cap.read()

            if ret: 
                # Recognize the pose
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frameRGB)
      
                if results.pose_landmarks:
                    # Read the value of the landmarks
                    lm = make_landmark_timestep(results)
                    lm_list.append(lm)
        
                    # Draw the landmarks on the frame
                    frame = draw_landmark_on_image(mpDraw, results, frame)
      
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()



# Save the landmarks to a csv file
def save_landmark_to_csv(label):
    df = pd.DataFrame(lm_list)
    df.to_csv(label + '.csv')
