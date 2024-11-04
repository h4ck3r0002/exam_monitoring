import tensorflow as tf
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Tải mô hình bằng TFSMLayer
model = tf.keras.layers.TFSMLayer("model.savedmodel", call_endpoint='serving_default')

# Tải nhãn lớp
class_names = open("labels.txt", "r").readlines()

# CAMERA có thể là 0 hoặc 1 tùy thuộc vào camera mặc định của máy tính
camera = cv2.VideoCapture(0)

while True:
    # Lấy hình ảnh từ webcam
    ret, image = camera.read()

    # Thay đổi kích thước hình ảnh thành 224x224 pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Hiển thị hình ảnh trong cửa sổ
    cv2.imshow("Webcam Image", image)

    # Chuyển đổi hình ảnh thành numpy array và định hình lại thành input cho mô hình
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Chuẩn hóa hình ảnh
    image = (image / 127.5) - 1

    # Dự đoán với mô hình
    prediction = model(image)

    # Kiểm tra cấu trúc đầu ra
    if isinstance(prediction, dict):
        output_key = list(prediction.keys())[0]  # Lấy tên của tensor đầu ra
        prediction = prediction[output_key]  # Lấy tensor từ từ điển

    # Tìm lớp dự đoán và điểm số độ tin cậy
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # In kết quả dự đoán và độ tin cậy
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Nhấn phím để thoát
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()
