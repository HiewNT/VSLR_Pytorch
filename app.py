# app.py
import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk
import cv2
from inference import get_args, inference  # Nhập hàm từ inference.py

# Lớp ứng dụng GUI
class SignLanguageApp:
    def __init__(self, root, args):
        self.root = root
        self.root.title("Nhận Diện Ngôn Ngữ Ký Hiệu")  # Tiêu đề cửa sổ
        self.root.geometry("800x700")
        self.running = True
        self.sentence = ""  # Văn bản hoàn chỉnh
        self.current_word = ""  # Từ hiện tại

        # Phần webcam trên cùng
        self.video_frame = tk.Frame(self.root, height=300)
        self.video_frame.pack(fill=tk.X, padx=10, pady=5)

        # Video từ webcam
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(side=tk.LEFT, padx=10, expand=True)

        # Phần văn bản nhận dạng ở giữa
        self.text_frame = tk.Frame(self.root)
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.text_label = tk.Label(self.text_frame, text="Ký tự nhận diện:", font=("Arial", 16))
        self.text_label.pack(anchor="n")

        self.recognized_text = tk.Label(self.text_frame, text="", font=("Arial", 20), wraplength=300, justify="left")
        self.recognized_text.pack(expand=True, anchor="w")

        # Phần nút ở dưới cùng
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Nút Xóa Từ Cuối
        self.delete_word_button = tk.Button(self.control_frame, text="Xóa Từ Cuối", bg="orange", fg="white", command=self.delete_word, font=("Arial", 16), width=15, height=1)
        self.delete_word_button.pack(side=tk.LEFT, padx=5)
        
        # Nút Xóa Tất Cả
        self.clear_button = tk.Button(self.control_frame, text="Xóa Tất Cả", bg="lightgray", command=self.clear_text, font=("Arial", 16), width=15, height=1)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Nút Lưu Vào File Văn Bản
        self.save_button = tk.Button(self.control_frame, text="Lưu Vào File Văn Bản", bg="green", fg="white", command=self.save_text, font=("Arial", 16), width=15, height=1)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Nút Thoát
        self.quit_button = tk.Button(self.control_frame, text="Thoát", bg="red", fg="white", command=self.on_closing, font=("Arial", 16), width=15, height=1)
        self.quit_button.pack(side=tk.LEFT, padx=5)

        # Chạy inference trong luồng riêng
        self.thread = Thread(target=inference, args=(args, self))
        self.thread.daemon = True
        self.thread.start()

        # Xử lý khi đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_video(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk

    def update_text(self, text):
        self.recognized_text.config(text=text)

    def clear_text(self):
        self.sentence = ""
        self.current_word = ""
        self.update_text("")

    def delete_word(self):
        if self.current_word:  # Nếu có từ hiện tại
            if len(self.current_word) > 0:
                self.current_word = self.current_word[:-1]  # Xóa ký tự cuối cùng của current_word
        elif self.sentence:  # Nếu không có current_word nhưng có sentence
            # Tách sentence thành danh sách từ
            words = self.sentence.split()
            if words:
                last_word = words[-1]  # Lấy từ cuối cùng
                if len(last_word) > 0:
                    last_word = last_word[:-1]  # Xóa ký tự cuối cùng của từ cuối
                    words[-1] = last_word  # Cập nhật từ cuối
                    self.sentence = " ".join(words)  # Gộp lại thành câu
        self.update_text((self.sentence + self.current_word).strip())

    def save_text(self):
        if not self.sentence.strip() and not self.current_word.strip():
            print("Không có văn bản để lưu!")
            return
        with open("recognized_text.txt", "w", encoding="utf-8") as file:
            file.write((self.sentence + self.current_word).strip())
        print("Đã lưu văn bản vào recognized_text.txt")

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    args = get_args()
    root = tk.Tk()
    app = SignLanguageApp(root, args)
    root.mainloop()