import cv2
import math
import numpy as np
import argparse
import time
import torch
from collections import Counter
from src.hand_tracking import handDetector
from src.classification import Classifier
from src.config import *
from src.utils import *

# Hàm lấy tham số dòng lệnh
def get_args():
    parser = argparse.ArgumentParser(description="Inferent Sign Language Detection")
    parser.add_argument("--image_size", "-i", type=int, default=300)
    parser.add_argument("--checkpoint", "-p", type=str, default="trained_models/last.pt")
    return parser.parse_args()

# Hàm lấy bounding box
def get_bounding_box(hand_landmarks, image_shape):
    h, w, _ = image_shape
    x_min = min([lm.x for lm in hand_landmarks]) * w
    x_max = max([lm.x for lm in hand_landmarks]) * w
    y_min = min([lm.y for lm in hand_landmarks]) * h
    y_max = max([lm.y for lm in hand_landmarks]) * h
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

# Hàm inference
def inference(args, app):
    prediction = []
    app.sentence = ""  # Văn bản hoàn chỉnh
    app.current_word = ""  # Từ hiện tại
    last_detection_time = time.time()
    no_hand_threshold = 1
    
    hand_detected_time = None
    recognition_started = False

    cap = cv2.VideoCapture(0)
    detector = handDetector()
    classifier = Classifier(args.checkpoint)

    while cap.isOpened() and app.running:
        ret, image = cap.read()
        if not ret or image is None:
            print("Không thể đọc camera")
            continue

        hands, image = detector.findHands(image)
        current_time = time.time()
        
        if hands:
            hand = hands[0]
            
            if hand_detected_time is None:
                hand_detected_time = current_time
                recognition_started = False
                print("Tay vừa xuất hiện, đợi 1 giây để nhận dạng...")

            time_elapsed = current_time - hand_detected_time
            if not recognition_started and (current_time - hand_detected_time) >= 1:
                recognition_started = True
                print("Bắt đầu nhận dạng sau 1 giây")
            
            if 'landmark' in hand:
                x, y, w, h = get_bounding_box(hand['landmark'], image.shape)
                h_img, w_img, _ = image.shape
                x1, y1 = max(0, x - 20), max(0, y - 20)
                x2, y2 = min(w_img, x + w + 20), min(h_img, y + h + 20)
                imgCrop = image[y1:y2, x1:x2]
                cv2.rectangle(image, (x1, y1), (x2, y2), MAGENTA, 4)
                
                if imgCrop.size > 0:
                    imgWhite = np.ones((args.image_size, args.image_size, 3), np.uint8) * 255
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        n = args.image_size / h
                        wCal = math.ceil(n * w)
                        imgResize = cv2.resize(imgCrop, (wCal, args.image_size))
                        if wCal > args.image_size:
                            imgResize = imgResize[:, :args.image_size]  # Cắt nếu vượt quá
                        wGap = math.ceil((args.image_size - wCal) / 2)
                        imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                    else:
                        n = args.image_size / w
                        hCal = math.ceil(n * h)
                        imgResize = cv2.resize(imgCrop, (args.image_size, hCal))
                        if hCal > args.image_size:
                            imgResize = imgResize[:args.image_size, :]  # Cắt nếu vượt quá
                        hGap = math.ceil((args.image_size - hCal) / 2)
                        imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                    results, index = classifier.prediction(imgWhite)
                    prediction.append(np.argmax(results))
                    probabilities = torch.softmax(torch.tensor(results), dim=0).numpy()

                    if most_common_value(prediction[-25:]) == np.argmax(probabilities) and max(probabilities) > 0.97:
                        raw_character = CLASSES[index]
                        character = special_characters_prediction(app.sentence + app.current_word, raw_character)
                        if character:
                            mapped_character = char_display_map.get(character, character)
                            if app.current_word:
                                last_char = app.current_word[-1]
                                special_character_replace = {
                                    "Mu": {"A": "Â", "O": "Ô", "E": "Ê"},
                                    "Munguoc": {"A": "Ă"},
                                    "Rau": {"O": "Ơ", "U": "Ư"}
                                }
                                special_character_block = {
                                    "Â": ["A", "Mu"], "Ê": ["E", "Mu"], "Ơ": ["O", "Rau"],
                                    "Ă": ["A", "Munguoc"], "Ư": ["U", "Rau"], "Ô": ["O", "Mu"]
                                }
                                valid_before = {
                                    "Mu": ["A", "E", "O"], "Munguoc": ["A"], "Rau": ["U", "O"]
                                }
                                if mapped_character == "Mu" and last_char not in valid_before["Mu"]:
                                    pass
                                elif mapped_character == "Munguoc" and last_char not in valid_before["Munguoc"]:
                                    pass
                                elif mapped_character == "Rau" and last_char not in valid_before["Rau"]:
                                    pass
                                elif last_char in special_character_block and raw_character in special_character_block[last_char]:
                                    pass
                                elif mapped_character in valid_before and last_char not in valid_before[mapped_character]:
                                    pass
                                elif raw_character in special_character_replace and last_char in special_character_replace[raw_character]:
                                    app.current_word = app.current_word[:-1] + special_character_replace[raw_character][last_char]
                                elif last_char == mapped_character:
                                    pass
                                elif last_char == "D" and mapped_character == "Đ":
                                    app.current_word = app.current_word[:-1] + "Đ"
                                elif last_char == "Đ" and mapped_character in ["D", "DD"]:
                                    pass
                                else:
                                    app.current_word += mapped_character
                            else:
                                if mapped_character not in ["Mu", "Munguoc", "Rau"]:
                                    app.current_word += mapped_character
                            last_detection_time = current_time

        else:
            if hand_detected_time is not None:
                hand_detected_time = None
                recognition_started = False
                print("Tay đã rời khỏi khung hình, reset trạng thái")
            if app.current_word and (current_time - last_detection_time) >= no_hand_threshold:
                app.sentence += app.current_word + " "
                app.current_word = ""

        # Tạo display_text để gửi sang GUI
        display_text = ""
        for char in app.sentence + app.current_word:
            display_text += char_display_map.get(char, char)
        
        # Cập nhật giao diện GUI
        app.update_video(image)
        app.update_text(display_text.strip())

    cap.release()
    print("Câu hoàn chỉnh:", display_text.strip())