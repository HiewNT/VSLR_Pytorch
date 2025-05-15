from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import torch
from src.hand_tracking import handDetector
from src.classification import Classifier
from src.config import *
from src.utils import *
import math
import time
from collections import Counter

app = Flask(__name__)
socketio = SocketIO(app)


# Khởi tạo các đối tượng
detector = handDetector()
classifier = Classifier("trained_models/last.pt")
image_size = 300

# Biến trạng thái
sentence = ""
current_word = ""
last_detection_time = time.time()
no_hand_threshold = 1
hand_detected_time = None
recognition_started = False
prediction = []

def get_bounding_box(hand_landmarks, image_shape):
    h, w, _ = image_shape
    x_min = min([lm.x for lm in hand_landmarks]) * w
    x_max = max([lm.x for lm in hand_landmarks]) * w
    y_min = min([lm.y for lm in hand_landmarks]) * h
    y_max = max([lm.y for lm in hand_landmarks]) * h
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

def process_frame(frame):
    global sentence, current_word, last_detection_time, hand_detected_time, recognition_started, prediction
    current_time = time.time()

    # Phát hiện tay
    hands, image = detector.findHands(frame)
    
    frame_to_send = None  # Biến để lưu ảnh gửi lại

    if hands:
        hand = hands[0]
        if hand_detected_time is None:
            hand_detected_time = current_time
            recognition_started = False

        time_elapsed = current_time - hand_detected_time
        if not recognition_started and time_elapsed >= 1:
            recognition_started = True

        if 'landmark' in hand:
            x, y, w, h = get_bounding_box(hand['landmark'], image.shape)
            h_img, w_img, _ = image.shape
            x1, y1 = max(0, x - 20), max(0, y - 20)
            x2, y2 = min(w_img, x + w + 20), min(h_img, y + h + 20)
            imgCrop = image[y1:y2, x1:x2]
            # Không vẽ hình chữ nhật lên image nữa vì chỉ cần ảnh tay

            if imgCrop.size > 0:
                imgWhite = np.ones((image_size, image_size, 3), np.uint8) * 255
                aspectRatio = h / w
                if aspectRatio > 1:
                    n = image_size / h
                    wCal = math.ceil(n * w)
                    imgResize = cv2.resize(imgCrop, (wCal, image_size))
                    if wCal > image_size:
                        imgResize = imgResize[:, :image_size]
                    wGap = math.ceil((image_size - wCal) / 2)
                    imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                else:
                    n = image_size / w
                    hCal = math.ceil(n * h)
                    imgResize = cv2.resize(imgCrop, (image_size, hCal))
                    if hCal > image_size:
                        imgResize = imgResize[:image_size, :]
                    hGap = math.ceil((image_size - hCal) / 2)
                    imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                results, index = classifier.prediction(imgWhite)
                prediction.append(np.argmax(results))
                probabilities = torch.softmax(torch.tensor(results), dim=0).numpy()

                if most_common_value(prediction[-25:]) == np.argmax(probabilities) and max(probabilities) > 0.98:
                    raw_character = CLASSES[index]
                    character = special_characters_prediction(sentence + current_word, raw_character)
                    if character:
                        mapped_character = char_display_map.get(character, character)
                        if current_word:
                            last_char = current_word[-1]
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
                            # Nếu từ trước là "Ư" và ký tự hiện tại là "U", cho phép nối tiếp thành "ƯU"
                            if last_char == "Ư" and mapped_character == "U":
                                current_word += "U"
                                
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
                                current_word = current_word[:-1] + special_character_replace[raw_character][last_char]
                            elif last_char == mapped_character:
                                pass
                            elif last_char == "D" and mapped_character == "Đ":
                                current_word = current_word[:-1] + "Đ"
                            elif last_char == "Đ" and mapped_character in ["D", "DD"]:
                                pass
                            else:
                                current_word += mapped_character
                        else:
                            if mapped_character not in ["Mu", "Munguoc", "Rau"]:
                                current_word += mapped_character
                        last_detection_time = current_time
                
                # Gửi ảnh vùng tay (imgWhite) thay vì toàn frame
                frame_to_send = imgWhite
    else:
        if hand_detected_time is not None:
            hand_detected_time = None
            recognition_started = False
        if current_word and (current_time - last_detection_time) >= no_hand_threshold:
            sentence += current_word + " "
            current_word = ""
        # Nếu không có tay, gửi một frame trắng
        frame_to_send = np.ones((image_size, image_size, 3), np.uint8) * 255

    # Tạo display_text
    display_text = ""
    for char in sentence + current_word:
        display_text += char_display_map.get(char, char)

    # Mã hóa frame để gửi lại (imgWhite hoặc frame trắng)
    _, buffer = cv2.imencode('.jpg', frame_to_send)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    return frame_encoded, display_text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def get_api_info():
    return jsonify({
        "message": "Danh sách API",
        "endpoints": {
            "/": "Trang chính",
            "/api": "Danh sách API",
            "/docs": "Tài liệu API Swagger",
        }
    })

@socketio.on('connect')
def handle_connect():
    emit('message', {'data': 'Connected to server'})

@socketio.on('video_frame')
def handle_video_frame(data):
    # Giải mã frame từ base64
    frame_data = base64.b64decode(data.split(',')[1])
    np_frame = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # Xử lý frame
    frame_encoded, recognized_text = process_frame(frame)

    # Gửi lại frame và kết quả nhận diện
    emit('response', {'frame': f'data:image/jpeg;base64,{frame_encoded}', 'text': recognized_text})
    
# Xử lý yêu cầu xóa từ client
@socketio.on('clear_text')
def handle_clear_text():
    global sentence, current_word
    sentence = ""  # Xóa câu hoàn chỉnh
    current_word = ""  # Xóa từ hiện tại
    emit('clear_response', {'text': ''})
    
@socketio.on('save_text')
def handle_save_text():
    global sentence, current_word
    try:
        full_text = sentence + current_word
        if not full_text.strip():
            emit('save_response', {
                'status': 'error',
                'message': 'Không có nội dung để lưu',
                'text': ''
            })
            return

        with open('saved_text.txt', 'a', encoding='utf-8') as f:
            f.write(full_text + '\n')

        # Gửi text về cho client để tạo file trên trình duyệt
        emit('save_response', {
            'status': 'success',
            'message': 'Đã lưu thành công',
            'text': full_text
        })

    except Exception as e:
        emit('save_response', {
            'status': 'error',
            'message': str(e),
            'text': ''
        })


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)