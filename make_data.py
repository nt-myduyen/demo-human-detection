import cv2
import mediapipe as mp
import pandas as pd
# sudo apt-get install libgtk2.0-dev pkg-config

# Read video from the default camera
cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
  print("Error opening video stream or file")

# set resolution, convert them from float to int
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, size)

# Initialize mediapipe lib
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "SITTING"
no_of_frames = 50

def make_landmark_timestep(results):
  print(results.pose_landmarks.landmark)
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
    print(id, lm)
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
  return frame

while len(lm_list) <= no_of_frames: 
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
      
      # write the frame to video 
      video.write(frame)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break
    
    
# Save the landmarks to a csv file
df = pd.DataFrame(lm_list)
df.to_csv(label + '.csv')


cap.release()
video.release()
cv2.destroyAllWindows()