from ultralytics import YOLO
import torch

# 1. Hardware Check (Safety First)
if torch.cuda.is_available():
    print(f"✅ Success! Training on: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU not found. Check your 'vision_worker' environment.")

# 2. Load the YOLO26 Nano model
# This is the 'Empty Brain' we are going to train.
model = YOLO("yolo26n.pt") 

# 3. Start Training
model.train(
    data="/home/linux/Documents/delete classes.yolov11/data.yaml", 
    epochs=50,       # 50 rounds of learning is perfect for a hackathon
    imgsz=416,       # We use 416 (instead of 640) to keep your 4GB GPU fast
    batch=16,        # How many images the AI looks at at once
    device=0,        # Force the GTX 1650 to work
    name="kitchen_safety_v1"
)
