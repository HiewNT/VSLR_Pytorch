import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None  # Lưu trữ kết quả từ Mediapipe

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        hands = []  # Danh sách chứa thông tin tay

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # Lấy bounding box và landmarks của bàn tay
                xList, yList = [], []
                h, w, _ = img.shape
                for lm in handLms.landmark:
                    xList.append(int(lm.x * w))
                    yList.append(int(lm.y * h))

                bbox = (min(xList), min(yList), max(xList) - min(xList), max(yList) - min(yList))
                hands.append({
                    "bbox": bbox,
                    "landmark": handLms.landmark  # Lưu danh sách các điểm mốc
                })

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return hands, img

    def findPosition(self, img, handNo=0, draw=True):
        """Trả về danh sách các điểm mốc của tay được chỉ định"""
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNo:
                handLms = self.results.multi_hand_landmarks[handNo]
                h, w, _ = img.shape
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Không thể truy cập camera.")
            break

        hands, img = detector.findHands(img)  # Lấy danh sách tay và ảnh đã vẽ
        lmList = detector.findPosition(img)   # Lấy danh sách các điểm mốc
        if lmList:
            print("Thumb tip position:", lmList[4])  # In vị trí đầu ngón tay cái (ID 4)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()