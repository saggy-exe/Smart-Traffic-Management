🚦 Smart Traffic Management System (Group 41)

A real-time vehicle detection, classification, and tracking system built using:

YOLOv5 (custom-trained on UVH-26 dataset)

DeepSORT (multi-object tracking)

The system processes traffic videos and outputs tracks with unique IDs, enabling traffic monitoring and analytics.

📌 Features

Detects vehicles across 14 classes from the UVH-26 dataset

Tracks each vehicle with persistent IDs

Works on real traffic video footage

Supports analytics such as movement patterns & vehicle count

🧠 Tech Stack

Python 3.10

YOLOv5 (Ultralytics)

DeepSORT

CUDA GPU acceleration

UVH-26 Vehicle Dataset

📦 Installation
git clone https://github.com/<your-username>/SmartTrafficSystem.git
cd SmartTrafficSystem
conda create -n traffic python=3.10
conda activate traffic
pip install -r requirements.txt
pip install deep-sort-realtime


Place the trained model:

SmartTrafficSystem/models/best.pt

▶️ Run Detection
python yolov5/detect.py --weights models/best.pt --source sample.jpg

▶️ Run Tracking (YOLO + DeepSORT)
python src/track_deepsort.py \
  --weights models/best.pt \
  --source input_video.mp4 \
  --output outputs/tracked_video.mp4

📁 Project Structure
SmartTrafficSystem/
├── src/                 # All custom scripts
├── models/              # best.pt (ignored in repo)
├── outputs/             # Result videos/images
├── docs/                # Presentation/document files
└── README.md

📊 Dataset

UVH-26 (IISc Bangalore)
14 selected vehicle categories, converted into YOLO format for training.

👥 Team (Group 41)

Detection & Model Training

Tracking & Integration

Documentation & Results
