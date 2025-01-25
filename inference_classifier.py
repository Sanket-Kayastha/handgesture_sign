import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from collections import deque


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=2,  
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.7)


labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'}


formed_word = ""
last_predicted_time = time.time()
prediction_buffer = deque(maxlen=10)

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        
        while len(data_aux) < 84:
            data_aux.extend([0, 0])  
        if len(data_aux) > 84:
            data_aux = data_aux[:84]  

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        predicted_character = ""
        
        if time.time() - last_predicted_time > 1:
            prediction = model.predict([np.asarray(data_aux)])
            prediction_buffer.append(prediction[0])
            
            
            if len(prediction_buffer) == prediction_buffer.maxlen:
                most_common_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                predicted_character = labels_dict[int(most_common_prediction)]
                formed_word += predicted_character
                prediction_buffer.clear()
                last_predicted_time = time.time()
                print(predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        if predicted_character:
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    
    cv2.putText(frame, 'Word: ' + formed_word, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('\r'):  
        formed_word = ""

cap.release()
cv2.destroyAllWindows()
