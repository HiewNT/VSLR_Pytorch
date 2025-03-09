import cv2
import os
import numpy as np
import math
import time
import hand_tracking  # Import module hand tracking

# Thông tin thư mục lưu trữ
DATA_DIR = "../dataset/alphabet"
CLASSES = ['A', 'B', 'C', 'D', 'DD', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'Mu', 'Munguoc', 
           'N', 'O', 'P', 'Q', 'R', 'Rau', 'S', 'T', 'U', 'V', 'X', 'Y']
IMG_SIZE = 300  # Kích thước ảnh đầu ra
OFFSET = 20  # Khoảng cách viền xung quanh bàn tay

# Tạo thư mục dữ liệu nếu chưa có
for label in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# Chọn lớp để thu thập
print("Danh sách lớp:", CLASSES)
selected_class = input("Nhập lớp cần thu thập : ").strip()

if selected_class not in CLASSES:
    print("Lớp không hợp lệ! Thoát chương trình.")
    exit()

# Kiểm tra số ảnh hiện có
class_dir = os.path.join(DATA_DIR, selected_class)
existing_images = len(os.listdir(class_dir))
print(f"Số ảnh hiện có: {existing_images}")

# Khởi động camera và detector
cap = cv2.VideoCapture(0)
detector = hand_tracking.handDetector(maxHands=1)  # Chỉ lấy 1 tay

while True:
    success, img = cap.read()
    if not success:
        print("Không thể truy cập camera.")
        break

    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Cắt vùng chứa bàn tay + padding
        x1, y1 = max(x - OFFSET, 0), max(y - OFFSET, 0)
        x2, y2 = min(x + w + OFFSET, img.shape[1]), min(y + h + OFFSET, img.shape[0])
        imgCrop = img[y1:y2, x1:x2]

        # Tạo ảnh trắng 300x300
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

        # Xác định tỉ lệ để resize
        aspectRatio = h / w
        if aspectRatio > 1:  # Cao hơn rộng
            scale = IMG_SIZE / h
            wResized = math.ceil(scale * w)
            imgResized = cv2.resize(imgCrop, (wResized, IMG_SIZE))
            wGap = math.ceil((IMG_SIZE - wResized) / 2)
            imgWhite[:, wGap:wGap + wResized] = imgResized
        else:  # Rộng hơn cao
            scale = IMG_SIZE / w
            hResized = math.ceil(scale * h)
            imgResized = cv2.resize(imgCrop, (IMG_SIZE, hResized))
            hGap = math.ceil((IMG_SIZE - hResized) / 2)
            imgWhite[hGap:hGap + hResized, :] = imgResized

        # Hiển thị ảnh
        cv2.imshow("Cropped", imgCrop)
        cv2.imshow("Processed", imgWhite)

    # Hiển thị số ảnh và hướng dẫn trên màn hình
    text = f"Class: {selected_class} | number of photos: {existing_images}"
    cv2.putText(img, f"Class: {selected_class} - Images: {existing_images}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Màu xanh dương

    cv2.putText(img, f"Press 's' to collect class '{selected_class}'", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Màu xanh lá

    cv2.putText(img, "Press 'q' to exit", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Màu vàng


    cv2.imshow("Camera", img)

    key = cv2.waitKey(1)
    if key == ord("s"):  # Nhấn 's' để lưu ảnh
        filename = os.path.join(class_dir, f"Image_{time.time()}.jpg")
        cv2.imwrite(filename, imgWhite)
        existing_images += 1
        print(f"Đã lưu ảnh: {filename}")
    elif key == ord("q"):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
