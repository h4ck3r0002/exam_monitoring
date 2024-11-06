import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Tải mô hình, scaler và label encoder
model = joblib.load('cheat_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
pca = joblib.load('pca.pkl')  # Tải PCA

def extract_hog_features(image):
    # Đảm bảo ảnh có kích thước 64x64
    image = cv2.resize(image, (64, 64))  # Điều chỉnh kích thước ảnh
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      block_norm='L2-Hys', visualize=True)
    return features

def predict_image(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hog_features = extract_hog_features(gray_image)

    hog_features = hog_features.reshape(1, -1)  # Reshape cho mô hình dự đoán
    hog_features = scaler.transform(hog_features)  # Chuẩn hóa dữ liệu
    hog_features = pca.transform(hog_features)  # Giảm chiều dữ liệu

    # Dự đoán
    prediction = model.predict(hog_features)
    probabilities = model.decision_function(hog_features)  # Lấy giá trị quyết định
    probabilities = 1 / (1 + np.exp(-probabilities))  # Chuyển đổi thành xác suất

    return label_encoder.inverse_transform(prediction)[0], probabilities[0]

# Đọc video từ camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, prob = predict_image(frame)  # Dự đoán
    # Hiển thị kết quả
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
