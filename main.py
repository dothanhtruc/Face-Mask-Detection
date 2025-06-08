import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
# from train_model import SimpleCNN  # hoặc copy trực tiếp class SimpleCNN nếu không dùng module


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Lớp 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Lớp 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Lớp 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Tính toán kích thước feature sau khi pooling 3 lần:
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Các tham số khớp với quá trình huấn luyện
input_size = 128
num_classes = 2
model_path = 'model/face_mask_detector.pth'

# Khởi tạo mô hình và load trọng số
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # chuyển mô hình sang chế độ eval

# Định nghĩa transform giống lúc training
data_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Hàm dự đoán nhãn cho một khuôn mặt
def predict_mask(face_img):
    # Chuyển đổi ảnh từ OpenCV (BGR) sang RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Chuyển ảnh PIL
    from PIL import Image
    face_pil = Image.fromarray(face_rgb)
    # Áp dụng transform
    input_tensor = data_transform(face_pil).unsqueeze(0)  # thêm chiều batch
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return pred.item()


# Hàm xử lý ảnh: phát hiện khuôn mặt và dự đoán mask/no mask
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        pred = predict_mask(face_img)
        label = "Mask" if pred == 0 else "No Mask"
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


# Hàm xử lý webcam
def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể truy cập webcam!")
        return
    print("Nhấn 'q' để thoát webcam.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_mask(frame)
        cv2.imshow("Face Mask Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Hàm xử lý ảnh tĩnh
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể mở ảnh từ đường dẫn: {image_path}")
        return
    frame = detect_mask(frame)
    cv2.imshow("Face Mask Detection - Ảnh tĩnh", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Hàm xử lý video từ file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video từ đường dẫn: {video_path}")
        return
    print("Nhấn 'q' để thoát chế độ video.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video hoặc lỗi đọc file!")
            break
        frame = detect_mask(frame)
        cv2.imshow("Face Mask Detection - Video", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Chọn chế độ:")
    print("1. Webcam")
    print("2. Nhập ảnh")
    print("3. Nhận video")
    option = input("Lựa chọn của bạn (1/2/3): ").strip()

    if option == '1':
        process_webcam()
    elif option == '2':
        image_path = input("Nhập đường dẫn ảnh: ").strip()
        process_image(image_path)
    elif option == '3':
        video_path = input("Nhập đường dẫn video: ").strip()
        process_video(video_path)
    else:
        print("Lựa chọn không hợp lệ!")
