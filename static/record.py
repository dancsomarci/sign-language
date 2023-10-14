import cv2
import mediapipe as mp
import time
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Landmark indicies to extract
idx = [i for i in range(21)]  # 21 landmarks
# 0 WRIST
# 1 THUMB_CNC
# 2 THUMB_MCP
# 3 THUMB_IP
# 4 THUMB_TIP
# 5 INDEX_FINGER_MCP
# 6 INDEX_FINGER_PIP
# 7 INDEX_FINGER_DIP
# 8 INDEX_FINGER_TIP
# 9 MIDDLE_FINGER_MCP
# 10 MIDDLE_FINGER_PIP
# 11 MIDDLE_FINGER_DIP
# 12 MIDDLE_FINGER_TIP
# 13 RING_FINGER_MCP
# 14 RING_FINGER_PIP
# 15 RING_FINGER_DIP
# 16 RING_FINGER_TIP
# 17 PINKY_MCP
# 18 PINKY_PIP
# 19 PINKY_DIP
# 20 PINKY_TIP


def retrieve_coordinates(landmark):
    return [landmark.x, landmark.y, landmark.z]


record_category = '9'
filename = "fehervari.csv"

recorded_frames = []

# Variables for counting fps
prev_frame = 0
new_frame = 0

# For webcam input:
cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # If 'r' is pressed -> 'r'ecord frame -> HOLD DOWN R FOR BEST PERFORMANCE
                if cv2.waitKey(33) == ord('r'):
                    temp = []  # For now only one hand
                    for i in idx:
                        temp.extend(retrieve_coordinates(
                            hand_landmarks.landmark[i]))
                    temp.append(record_category)
                    recorded_frames.append(temp)

        # Flipping image for selfie view
        image = cv2.flip(image, 1)

        # Calculating fps
        new_frame = time.time()
        fps = int(1/(new_frame-prev_frame))
        image = cv2.putText(image, "fps: " + str(fps), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 150, 0), 2, cv2.LINE_AA)
        prev_frame = new_frame

        # r = 100
        # cv2.rectangle(image, (int(width/2 - r), int(height/2 - r)),
        #               (int(width/2 + r), int(height/2 + r)), (255, 100, 0), 2)

        cv2.imshow('Frame', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release webcam
cap.release()

# Save frames
if len(recorded_frames) > 0:
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(recorded_frames)