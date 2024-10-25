import cv2
import numpy as np
from channels.generic.websocket import WebsocketConsumer

class VideoStreamConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        print("Kết nối WebSocket thành công.")

    def disconnect(self, close_code):
        print("Đóng kết nối WebSocket.")

    def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Xử lý từng frame từ video stream
            np_data = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            # Áp dụng các thuật toán phát hiện gian lận
            cheating_detected = self.detect_cheating(frame)
            if cheating_detected:
                # Gửi cảnh báo về client nếu phát hiện gian lận
                self.send(text_data="Cheating detected!")

    def detect_cheating(self, frame):
        # Thay thế với các thuật toán phát hiện gian lận
        # Ví dụ: phát hiện nhiều khuôn mặt trong khung hình
        # hoặc các hành vi khác
        # Trả về True nếu phát hiện gian lận, ngược lại False
        return False
