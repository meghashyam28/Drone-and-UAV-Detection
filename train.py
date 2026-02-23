from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/drone.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        optimizer="Adam",
        lr0=0.001,
        project="models",
        name="drone_uav_detector",
        verbose=True
    )

    print("Training finished. Best model saved.")

if __name__ == "__main__":
    train_model()
