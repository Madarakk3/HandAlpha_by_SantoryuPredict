
import argparse
import copy
import time
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf


class CvFpsCalc:
    def __init__(self, buffer_len=10):
        self._times = deque(maxlen=buffer_len)

    def get(self):
        now = time.time()
        self._times.append(now)
        if len(self._times) >= 2:
            period = self._times[-1] - self._times[0]
            fps = (len(self._times) - 1) / period if period > 0 else 0.0
        else:
            fps = 0.0
        return fps


# ----------------------------------
# Здесь замените "asl_model.h5" на путь к вашей модели,
# если у вас другая модель, или пропустите этот блок, если тестируете только зеркалку.
# ----------------------------------
model = tf.keras.models.load_model("asl_model.h5")
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args, _ = parser.parse_known_args()
    return args


def extract_and_preprocess_roi(image_bgr, brect):
    x1, y1, x2, y2 = brect
    h, w, _ = image_bgr.shape
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    roi = image_bgr[y1:y2, x1:x2]
    roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    roi_resized = cv.resize(roi_gray, (60, 60))
    roi_norm = roi_resized.astype("float32") / 255.0
    img_tensor = np.expand_dims(np.expand_dims(roi_norm, axis=-1), axis=0)
    return img_tensor


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)
        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Рисуем сами точки поверх линий
    for index, landmark in enumerate(landmark_point):
        radius = 5
        if index in (4, 8, 12, 16, 20):
            radius = 8
        cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
               cv.LINE_AA)
    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        lx = min(int(landmark.x * image_width), image_width - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, np.array([[lx, ly]]), axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        lx = min(int(landmark.x * image_width), image_width - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])
    return landmark_point


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    min_det = args.min_detection_confidence
    min_track = args.min_tracking_confidence
... 
...     use_brect = True
... 
...     cap = cv.VideoCapture(cap_device)
...     cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
...     cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
... 
...     mp_hands = mp.solutions.hands
...     hands = mp_hands.Hands(
...         static_image_mode=False,
...         max_num_hands=1,
...         min_detection_confidence=min_det,
...         min_tracking_confidence=min_track,
...     )
... 
...     cvFpsCalc = CvFpsCalc(buffer_len=10)
...     point_history = deque(maxlen=16)
... 
...     while True:
...         fps = cvFpsCalc.get()
...         key = cv.waitKey(10)
...         if key == 27:
...             break
... 
...         # Сразу читаем кадр и зеркалим (горизонтально)
...         ret, image = cap.read()
...         if not ret:
...             break
...         image = cv.flip(image, 1)
... 
...         # Детектируем на уже зеркальном кадре
...         image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
...         image_rgb.flags.writeable = False
...         results = hands.process(image_rgb)
...         image_rgb.flags.writeable = True
... 
...         debug_image = copy.deepcopy(image)
... 
...         if results.multi_hand_landmarks is not None:
...             for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
...                                                   results.multi_handedness):
...                 brect = calc_bounding_rect(image, hand_landmarks)
...                 landmark_list = calc_landmark_list(image, hand_landmarks)
... 
...                 # Классификация ROI (по зеркальному кадру)
...                 img_tensor = extract_and_preprocess_roi(image, brect)
...                 if img_tensor is not None:
...                     preds = model.predict(img_tensor)
...                     idx = np.argmax(preds[0])
...                     predicted_label = class_names[idx]
...                 else:
...                     predicted_label = ""
... 
...                 point_history.append([0, 0])
... 
...                 # Рисуем bounding box, скелет и текст поверх зеркального кадра
...                 debug_image = draw_bounding_rect(use_brect, debug_image, brect)
...                 debug_image = draw_landmarks(debug_image, landmark_list)
...                 debug_image = draw_info_text(
...                     debug_image,
...                     brect,
...                     handedness,
...                     predicted_label,
...                     ""
...                 )
...         else:
...             point_history.append([0, 0])
... 
...         debug_image = draw_point_history(debug_image, point_history)
...         debug_image = draw_info(debug_image, fps, 0, -1)
... 
...         cv.imshow('Hand Gesture Recognition (Mirrored)', debug_image)
... 
...     cap.release()
...     cv.destroyAllWindows()
... 
... 
... if __name__ == '__main__':
...     main()
