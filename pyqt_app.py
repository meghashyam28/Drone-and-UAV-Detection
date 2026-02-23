import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO


MODEL_PATH = r"C:\Users\vikra\OneDrive\Desktop\Drone and UAV Detection\models\drone_uav_detector3\weights\best.pt"


class DroneDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drone & UAV Detection System")
        self.setGeometry(200, 100, 1000, 700)

        # YOLO model
        self.model = YOLO(MODEL_PATH)

        # Video
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # UI Elements
        self.video_label = QLabel("Upload a video to start detection")
        self.video_label.setFixedSize(960, 540)
        self.video_label.setStyleSheet("border: 2px solid black")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.select_btn = QPushButton("Select Video")
        self.start_btn = QPushButton("Start Detection")
        self.stop_btn = QPushButton("Stop")

        self.select_btn.clicked.connect(self.select_video)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        # Layouts
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.video_label)

        self.setLayout(main_layout)

        self.video_path = None

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.video_label.setText("Video selected. Click Start Detection.")

    def start_detection(self):
        if not self.video_path:
            self.video_label.setText("Please select a video first.")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        self.timer.start(30)  # ~33 FPS

    def stop_detection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("Detection stopped.")

    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        # 🔥 SPEED OPTIMIZATION (VERY IMPORTANT)
        frame = cv2.resize(frame, (640, 360))

        # YOLO inference
        results = self.model(frame)
        annotated = results[0].plot()

        # Convert to Qt image
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(qt_img))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneDetectionApp()
    window.show()
    sys.exit(app.exec_())
