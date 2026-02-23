import cv2
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

MODEL_PATH = "models/drone_uav_detector3/weights/best.pt"

class DroneUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone & UAV Detection System")

        self.model = YOLO(MODEL_PATH)
        self.cap = None
        self.running = False

        # Video display area
        self.video_label = Label(root)
        self.video_label.pack()

        # Button frame
        btn_frame = Frame(root)
        btn_frame.pack()

        self.start_btn = Button(btn_frame, text="Start Detection", command=self.start_detection, width=20, bg="green", fg="white")
        self.start_btn.grid(row=0, column=0, padx=10, pady=10)

        self.stop_btn = Button(btn_frame, text="Stop Detection", command=self.stop_detection, width=20, bg="red", fg="white")
        self.stop_btn.grid(row=0, column=1, padx=10, pady=10)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture("videos/test_video.mp4")
            self.update_frame()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def update_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret:
                results = self.model(frame, stream=True)

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = f"UAV {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Convert to Tkinter image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(frame)

                self.video_label.img_tk = img_tk
                self.video_label.config(image=img_tk)

            else:
                self.stop_detection()

        self.root.after(10, self.update_frame)


# Run UI
root = Tk()
app = DroneUI(root)
root.mainloop()
