import argparse
from collections import deque
from collections import Counter

import cv2
import mediapipe as mp
import tensorflow as tf

FONT_COLOR = (0, 0, 0) # BGR format
FONT_SIZE = 2
FONT_WEIGHT = 3

################################
# Video input
################################
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def video_loop_mp_hands(source, process_data_func):
    video = cv2.VideoCapture(source)
    with mp_hands.Hands(static_image_mode=True, model_complexity=1,max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
        while video.isOpened():
            _, image = video.read()

            if image is None:
                break

            image. flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks: # 1 hand only
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
    
                    # Get bounding box for the hand
                    lms = hand_landmarks.landmark
                    x_coords = [lm.x for lm in lms]
                    y_coords = [lm.y for lm in lms]
                    min_x, max_x = int(min(x_coords) * image.shape[1]), int(max(x_coords) * image.shape[1])
                    min_y, max_y = int(min(y_coords) * image.shape[0]), int(max(y_coords) * image.shape[0])

                    # Save for later
                    old_min_x = min_x
                    old_min_y = min_y

                    # Add margin proportional to width and height
                    width = max_x-min_x
                    height = max_y-min_y
                    expansion_rate = 0.35
                    min_y = max(int(min_y-expansion_rate*height), 0)
                    max_y = int(max_y+expansion_rate*height)
                    min_x = max(int(min_x-expansion_rate*width), 0)
                    max_x = int(max_x+expansion_rate*width)

                    # Crop hand image so that far away hands appear similar to images from dataset
                    hand_image = image[min_y:max_y, min_x:max_x]

                    # Collect landmark coordinates
                    data = []
                    # If the cropped hand is not too small to detect use landmarks from that image
                    if width > 0 and height > 0:
                        zoomed_result = hands.process(hand_image)
                        if zoomed_result.multi_hand_landmarks:
                            # Gather landmark coordinates
                            for lm in zoomed_result.multi_hand_landmarks[0].landmark:
                                data.extend([lm.x, lm.y, lm.z])

                    # As a fallback use the data from original image. (This won't be as accurate.)
                    if len(data) == 0:
                        for lm in lms:
                            data.extend([lm.x, lm.y, lm.z])

                    # Get prediction and probability from function
                    pred, _prob = process_data_func(data)

                    image = cv2.flip(image, 1)
                    # Display prediction and probability near the bounding box
                    label = pred if pred else ""
                    cv2.putText(image, label.upper(), (image.shape[1] - old_min_x, old_min_y-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_WEIGHT, cv2.LINE_AA)
            else:
                image = cv2.flip(image, 1)
            
            cv2.imshow("Asl fingerspelling demo", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    video.release()
        
################################
# MODEL
################################
class BaseModel:
    def __init__(self, model):
        self.model = model

    def process_frame(self, frame):
        res = self.model.predict(frame)
        pred = res["result"].numpy().decode("utf-8")
        prob = res["confidence"].numpy()
        return pred, prob

class ContinuousModel:
    def __init__(self, base_model, buffer_size = 20, confidence=0.7):
        self.base_model = base_model
        self.last_pred = None
        self.word = ""
        self.buffer = deque(maxlen=buffer_size)
        self.confidence_threshold = int(buffer_size * confidence)

    def process_frame(self, frame):
        pred, _prob = self.base_model.process_frame(frame)
        self.buffer.append(pred)
        buffered_pred, count = Counter(self.buffer).most_common(1)[0]
        if count >= self.confidence_threshold and self.last_pred != buffered_pred:
            self.last_pred = pred
            self.word += pred
        return self.word, 0.

if __name__ == "__main__":
    # parse cli arguments
    parser = argparse.ArgumentParser(description="American sign language fingerspelling demo, from static image data.")
    parser.add_argument(
        "-s",
        "--source",
        default=0,
        help="Numbers to select camera device, path to work from video source.",
    )
    parser.add_argument(
        "-cw",
        "--connectwords",
        action="store_true",
        default=False,
        help="Connect predictions to form words.",
    )
    args = parser.parse_args()

    base_model = tf.saved_model.load("static_fingerspelling_demo_model")
    model = BaseModel(base_model)

    if args.connectwords:
        model = ContinuousModel(model)

    video_loop_mp_hands(args.source, lambda data: model.process_frame(data))
