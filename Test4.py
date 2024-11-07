import cv2
import numpy as np
import joblib
import mediapipe as mp

# Tải mô hình, scaler và label encoder với tên file mới
model = joblib.load('cheat_detection_pose_model.pkl')
scaler = joblib.load('scaler_pose.pkl')
label_encoder = joblib.load('label_encoder_pose.pkl')

# Khởi tạo mô hình Pose của MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hàm trích xuất các keypoints của pose
def extract_pose_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints)
    else:
        return None

def predict_image(frame):
    keypoints = extract_pose_keypoints(frame)
    if keypoints is not None:
        keypoints = keypoints.reshape(1, -1)  # Định dạng lại để phù hợp với đầu vào của mô hình
        keypoints = scaler.transform(keypoints)  # Chuẩn hóa

        # Dự đoán
        prediction = model.predict(keypoints)
        probabilities = model.decision_function(keypoints)  # Lấy giá trị quyết định
        probabilities = 1 / (1 + np.exp(-probabilities))  # Chuyển đổi thành xác suất

        return label_encoder.inverse_transform(prediction)[0], probabilities[0]
    else:
        return "No Pose Detected", None

# Đọc video từ camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, prob = predict_image(frame)  # Dự đoán
    if prob is not None:
        prob_cheat = prob if label == "cheat" else 1 - prob
        prob_normal = 1 - prob_cheat

        # Hiển thị kết quả
        cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(frame, f'Probability Cheat: {prob_cheat:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #cv2.putText(frame, f'Probability Normal: {prob_normal:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Prediction: Cheat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
