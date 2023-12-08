import os 
import sys

import cv2
import tensorflow as tf
import mediapipe as mp

WEBCAM = 0
FONT_COLOR = (0, 0, 0) # BGR format
FONT_SIZE = 2
FONT_WEIGHT = 3

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_image(image, results):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
    )
    return image

def video_loop_mp_hands(source, process_result_func):
    video = cv2.VideoCapture(source)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while video.isOpened():
            _, image = video.read()

            if image is None:
                break

            image. flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            pred = process_result_func(results)

            # Draw landmark annotation on the image.
            image = draw_landmarks_on_image(image, results)
            if pred:
                cv2.putText(image, pred.upper(), (image.shape[1] - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_WEIGHT, cv2.LINE_AA)
            
            cv2.imshow("Asl fingerspelling demo", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    video.release()

LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

# Note that currently the formatter only supports pose and hand landmarks not face landmarks
class ModelInputFormatter:
    def __init__(self, model):
        self.required_landmarks = [bytes.decode("utf-8") for bytes in model.info().numpy()]

    def get_model_input(self, mp_holistic_result):
        (rx, ry, rz), (lx, ly, lz), (px, py, pz) = self._extract_from_result(mp_holistic_result)

        mapped_list = []
        for item in self.required_landmarks:
            parts = item.split('_')
            idx = int(parts[-1]) # Extract the index
        
            if parts[0] == 'x':
                if 'right_hand' in item:
                    mapped_list.append(rx[idx])
                elif 'left_hand' in item:
                    mapped_list.append(lx[idx])
                elif 'pose' in item:
                    mapped_list.append(px[idx])
        
            elif parts[0] == 'y':
                if 'right_hand' in item:
                    mapped_list.append(ry[idx])
                elif 'left_hand' in item:
                    mapped_list.append(ly[idx])
                elif 'pose' in item:
                    mapped_list.append(py[idx])
        
            elif parts[0] == 'z':
                if 'right_hand' in item:
                    mapped_list.append(rz[idx])
                elif 'left_hand' in item:
                    mapped_list.append(lz[idx])
                elif 'pose' in item:
                    mapped_list.append(pz[idx])

        return mapped_list

    def _extract_from_result(self, res):
        # Extract specific pose landmarks if available
        px = []
        py = []
        pz = []
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark:
                px.append(lm.x)
                py.append(lm.y)
                pz.append(lm.z)
        else:
            px = [0.0]*len(POSE)
            py = [0.0]*len(POSE)
            pz = [0.0]*len(POSE)
    
        # Extract left hand landmarks if available
        lx = []
        ly = []
        lz = []
        if res.left_hand_landmarks:
            for lm in res.left_hand_landmarks.landmark:
                lx.append(lm.x)
                ly.append(lm.y)
                lz.append(lm.z)
        else:
            lx = [0.0]*21
            ly = [0.0]*21
            lz = [0.0]*21
    
        # Extract right hand landmarks if available
        rx = []
        ry = []
        rz = []
        if res.right_hand_landmarks:
            for lm in res.right_hand_landmarks.landmark:
                rx.append(lm.x)
                ry.append(lm.y)
                rz.append(lm.z)
        else:
            rx = [0.0]*21
            ry = [0.0]*21
            rz = [0.0]*21
    
        return (rx, ry, rz), (lx, ly, lz), (px, py, pz)
    
class NonContinuousRecognitionModel:
    def __init__(self, model, max_out_length=31, confidence_threshold=0.2):
        self.model = model
        self.formatter = ModelInputFormatter(self.model)

        self.max_out_length = max_out_length
        # Only predictions with a higher confidence count as a predicted character
        self.confidence_threshold = confidence_threshold
        self.input = []

    def reset_buffer(self):
        self.input.clear()

    def translate_buffer(self, reset_buffer=False):
        res = None
        if len(self.input) > 0:
            res = self._generate_with_confidence()
            
        if reset_buffer:
            self.reset_buffer()
            
        return res

    def process_frame(self, mp_holistic_result):
        selected_landmarks_for_model = self.formatter.get_model_input(mp_holistic_result)
        self.input.append(selected_landmarks_for_model)

    def _generate_with_confidence(self):
        ctx = "<"
        for i in range(self.max_out_length):
            res = self.model.predict(self.input, ctx)
            res_char = res["result"].numpy().decode("utf-8")
            prob = res["confidence"].numpy()
            if prob > self.confidence_threshold:
                ctx += res_char
                if res_char == ">":
                    break
        return ctx

if __name__ == "__main__":
    loaded_model = tf.saved_model.load(os.path.join("trained_models", "Transformer_no_pos_embed", "transformer_seq2seq_saved_model"))
    fs_model = NonContinuousRecognitionModel(loaded_model, max_out_length=31, confidence_threshold=0.0)
    video_loop_mp_hands(os.path.join("test_videos", sys.argv[1]), lambda data: fs_model.process_frame(data))
    print(fs_model.translate_buffer())