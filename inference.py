import cv2
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import time
import torch
from collections import Counter
from src.hand_tracking import handDetector
from src.classification import Classifier
from src.config import *
from src.utils import *

def get_args():
    parser = argparse.ArgumentParser(description="Inferent Sign Language Detection")
    parser.add_argument("--image_size", "-i", type=int, default=300)
    parser.add_argument("--checkpoint", "-p", type=str, default="trained_models/last.pt")
    return parser.parse_args()

def get_bounding_box(hand_landmarks, image_shape):
    h, w, _ = image_shape
    x_min = min([lm.x for lm in hand_landmarks]) * w
    x_max = max([lm.x for lm in hand_landmarks]) * w
    y_min = min([lm.y for lm in hand_landmarks]) * h
    y_max = max([lm.y for lm in hand_landmarks]) * h
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

def inference(args):
    prediction = []
    sentence = ""  # Văn bản hoàn chỉnh
    current_word = ""  # Từ hiện tại đang xây dựng
    last_detection_time = time.time()  # Thời gian lần cuối phát hiện bàn tay
    no_hand_threshold = 1  # Ngưỡng thời gian để kết thúc từ (0.5 - 1 giây)
    
    hand_detected_time = None  # Thời gian tay xuất hiện lần đầu
    recognition_started = False  # Trạng thái bắt đầu nhận dạng

    cap = cv2.VideoCapture(0)
    detector = handDetector()
    classifier = Classifier(args.checkpoint)

    while cap.isOpened():
        ret, image = cap.read()
        if not ret or image is None:
            print("Không thể đọc camera")
            continue

        hands, image = detector.findHands(image)
        
        # Chuyển ảnh sang PIL để vẽ văn bản
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        font_path = "Roboto-VariableFont_wdth,wght.ttf"
        font = ImageFont.truetype(font_path, 40)

        current_time = time.time()
        
        if hands:
            hand = hands[0]
            
            if hand_detected_time is None:
                hand_detected_time = current_time
                recognition_started = False
                print("Tay vừa xuất hiện, đợi 1 giây để nhận dạng...")

            # Hiển thị thời gian đếm ngược (tùy chọn)
            time_elapsed = current_time - hand_detected_time
            if time_elapsed < 1:
                countdown = int(1 - time_elapsed) + 0.5  # Đếm ngược: 2, 1
                draw.text((50, 50), f"Đợi: {countdown}s", font=font, fill=(255, 255, 0))

            # Kiểm tra nếu đã đợi đủ 1 giây
            if not recognition_started and (current_time - hand_detected_time) >= 1:
                recognition_started = True
                print("Bắt đầu nhận dạng sau 1 giây")
            
            if 'landmark' in hand:
                x, y, w, h = get_bounding_box(hand['landmark'], image.shape)

                h_img, w_img, _ = image.shape
                x1, y1 = max(0, x - 20), max(0, y - 20)
                x2, y2 = min(w_img, x + w + 20), min(h_img, y + h + 20)
                imgCrop = image[y1:y2, x1:x2]
                
                # Vẽ khung bao quanh tay (tùy chọn)
                cv2.rectangle(image, (x1, y1), (x2, y2), MAGENTA, 4)
                
                if imgCrop.size > 0:
                    imgWhite = np.ones((args.image_size, args.image_size, 3), np.uint8) * 255

                    aspectRatio = h / w
                    if aspectRatio > 1:
                        n = args.image_size / h
                        wCal = math.ceil(n * w)
                        imgResize = cv2.resize(imgCrop, (wCal, args.image_size))
                        wGap = math.ceil((args.image_size - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        n = args.image_size / w
                        hCal = math.ceil(n * h)
                        imgResize = cv2.resize(imgCrop, (args.image_size, hCal))
                        hGap = math.ceil((args.image_size - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    results, index = classifier.prediction(imgWhite)
                    prediction.append(np.argmax(results))
                    
                    # Chuyển logits sang xác suất
                    probabilities = torch.softmax(torch.tensor(results), dim=0).numpy()

                    # Kiểm tra điều kiện với xác suất > 98%
                    if most_common_value(prediction[-25:]) == np.argmax(probabilities) and max(probabilities) > 0.97:  # Chỉ nhận nếu xác suất > 98%
                        
                        raw_character = CLASSES[index]  # Ký tự nhận diện ban đầu
                        character = special_characters_prediction(sentence + current_word, raw_character)
                        
                        if character:  # Kiểm tra nếu không phải None thì mới xử lý
                            mapped_character = char_display_map.get(character, character)  # Ánh xạ ngay
                            
                            # Các ký tự đặc biệt và quy tắc thay thế
                            special_character_replace = {
                                "Mu": {"A": "Â", "O": "Ô", "E": "Ê"},
                                "Munguoc": {"A": "Ă"},
                                "Rau": {"O": "Ơ", "U": "Ư"}
                            }

                            # Chặn ký tự gốc nếu đã có ký tự đặc biệt
                            special_character_block = {
                                "Â": ["A", "Mu"], 
                                "Ê": ["E", "Mu"], 
                                "Ơ": ["O", "Rau"], 
                                "Ă": ["A", "Munguoc"], 
                                "Ư": ["U", "Rau"], 
                                "Ô": ["O", "Mu"]
                            }
                            
                            # Danh sách các ký tự hợp lệ trước Mu, Munguoc, Rau
                            valid_before = {
                                "Mu": ["A", "E", "O"],
                                "Munguoc": ["A"],
                                "Rau": ["U", "O"]
                            }
                            
                            if current_word:
                                last_char = current_word[-1]

                                # Nếu Mu, Rau, Munguoc không có ký tự trước hợp lệ, thì bỏ qua
                                if mapped_character == "Mu" and last_char not in ['A', 'E', 'O']:
                                    pass  # Bỏ qua nếu "Mu" không có ký tự hợp lệ trước đó
                                if mapped_character == "Munguoc" and last_char not in ['A']:
                                    pass  # Bỏ qua nếu "Munguoc" không có "A" trước đó
                                if mapped_character == "Rau" and last_char not in ['U', 'O']:
                                    pass  # Bỏ qua nếu "Rau" không có "U" hoặc "O" trước đó

                                # Nếu last_char là ký tự đặc biệt và ký tự hiện tại thuộc nhóm bị chặn → bỏ qua
                                if last_char in special_character_block and raw_character in special_character_block[last_char]:
                                    pass  # Không thêm ký tự bị chặn
                                
                                # Nếu là Mu, Munguoc, Rau nhưng không có ký tự trước hợp lệ → bỏ qua
                                elif mapped_character in valid_before and last_char not in valid_before[mapped_character]:
                                    pass  # Bỏ qua nếu trước đó không hợp lệ
                                
                                # Nếu gặp dấu phụ (Mu, Rau, Munguoc), thay thế ký tự gốc
                                elif raw_character in special_character_replace and last_char in special_character_replace[raw_character]:
                                    current_word = current_word[:-1] + special_character_replace[raw_character][last_char]

                                # Nếu current_word không phải ký tự đặc biệt, không cho lặp lại chính nó
                                elif last_char == mapped_character:
                                    pass  # Không thêm lại chính ký tự đó
                                

                                # Nếu có D rồi mà chữ tiếp theo là DD thì thay D thành Đ
                                elif current_word and current_word[-1] == "D" and mapped_character == "Đ":
                                    current_word = current_word[:-1] + "Đ"

                                # Nếu đã có Đ rồi mà chữ tiếp theo là D hoặc DD thì không thêm
                                elif current_word and current_word[-1] == "Đ" and mapped_character in ["D", "DD"]:
                                    pass  # Không thêm ký tự

                                else:
                                    current_word += mapped_character  # Thêm ký tự mới nếu không bị chặn

                            else:
                                # Nếu current_word rỗng, chỉ thêm ký tự hợp lệ (loại bỏ Mu, Munguoc, Rau)
                                if mapped_character not in ["Mu", "Munguoc", "Rau"]:
                                    current_word += mapped_character
                                                                
                        last_detection_time = current_time  # Cập nhật thời gian phát hiện

        else:
            # Nếu tay biến mất, reset trạng thái để lần sau đợi lại 2 giây
            if hand_detected_time is not None:
                hand_detected_time = None
                recognition_started = False
                print("Tay đã rời khỏi khung hình, reset trạng thái")
                
            # Nếu không phát hiện bàn tay, kiểm tra thời gian
            if current_word and (current_time - last_detection_time) >= no_hand_threshold:
                sentence += current_word + " "  # Kết thúc từ, thêm khoảng trắng
                current_word = ""  # Reset từ hiện tại

        # Hiển thị văn bản ở chính giữa trên cùng với ánh xạ ký tự đặc biệt
        display_text = ""
        for char in sentence + current_word:
            display_text += char_display_map.get(char, char)  # Ánh xạ ký tự đặc biệt
        
        text_size = draw.textbbox((0, 0), display_text, font=font)
        text_width = text_size[2] - text_size[0]
        window_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        text_x = (window_width - text_width) // 2  # Căn giữa
        text_y = 10  # Gần mép trên
        draw.text((text_x, text_y), display_text, font=font, fill=(255, 255, 255))

        # Chuyển lại sang OpenCV
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Image", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Câu hoàn chỉnh:", display_text.strip())

if __name__ == '__main__':
    args = get_args()
    inference(args)