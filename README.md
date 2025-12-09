🚦 Smart Traffic Management System – Group 41
Real-time Vehicle Detection, Classification, and Tracking using YOLOv5 + DeepSORT

B.Tech CSE – Academy Of Technology • MAKAUT

📌 Overview

This project implements an intelligent Smart Traffic Management System capable of:

Detecting vehicles in real-time

Classifying them into 14 UVH-26 categories

Tracking each vehicle with unique IDs across frames

Using these insights for traffic analytics and congestion monitoring

It uses the UVH-26 dataset (IISc Bangalore) and a custom-trained YOLOv5 model combined with DeepSORT for object tracking.

This system is designed to support smart-city traffic applications such as automated traffic rule enforcement, congestion analysis, and vehicle flow monitoring.

🎯 Objectives

Train a custom YOLOv5 model on UVH-26 for accurate vehicle detection

Track multiple vehicles in real-time using DeepSORT

Process real-world traffic video footage

Produce analytics-ready output videos with bounding boxes + track IDs

🧠 Architecture
1. YOLOv5 (Detection & Classification)

14 classes

UVH-26 dataset converted to YOLO format

Trained with CUDA-enabled GPU

Outputs bounding boxes + class predictions

2. DeepSORT (Tracking)

Uses YOLOv5 detections as input

Assigns persistent IDs to each moving vehicle

Generates trajectory & motion information

3. Smart Traffic Pipeline
Video → YOLOv5 Detection → DeepSORT Tracker → ID’d Vehicles → Output Video

🗂️ Folder Structure
SmartTrafficSystem/
│
├── src/
│   ├── track_deepsort.py         # Tracking pipeline
│   ├── repack_uvh26_yolo.py      # Dataset subfolder extractor
│   ├── coco2yolo.py              # Annotation converter
│   ├── check_labels.py           # Validation scripts
│   ├── ... (support scripts)
│
├── models/
│   └── best.pt                   # (ignored in git) trained YOLO model
│
├── docs/
│   ├── project_presentation.pdf
│   └── pipeline_diagram.png
│
├── outputs/
│   ├── test_video_results.mp4
│   └── tracking_sample.jpg
│
├── .gitignore
└── README.md

📦 Setup Instructions
1. Clone the repository
git clone https://github.com/<your-username>/SmartTrafficSystem.git
cd SmartTrafficSystem

2. Create Conda environment
conda create -n traffic python=3.10
conda activate traffic

3. Install dependencies
pip install -r requirements.txt
pip install deep-sort-realtime

4. Download trained YOLO model

Place the file here:

SmartTrafficSystem/models/best.pt


(Model file is ignored in GitHub due to large size.)

🚗 Running Detection

To run object detection on an image:

python yolov5/detect.py \
  --weights models/best.pt \
  --source sample.jpg \
  --img 640 \
  --conf 0.25

🚘 Running Tracking (YOLO + DeepSORT)

To run tracking on a video:

python src/track_deepsort.py \
  --weights models/best.pt \
  --source input_video.mp4 \
  --output outputs/tracked_video.mp4


This generates a video with:

Bounding boxes

Class names

Persistent tracking IDs

📊 Results
Custom YOLOv5 Model

Successfully trained on UVH-26 dataset

Model works on real traffic footage

Evaluated using real-world test videos

Tracking Output (DeepSORT)

Vehicle IDs maintained consistently

Smooth multi-object tracking

Suitable for analytics like counting & congestion estimation

Snapshots from output video:

(Add your screenshots here)

docs/results_frame_1.png
docs/results_frame_2.png

🧪 Dataset Used

UVH-26 (Urban Vehicle Hood Dataset)
Source: IISc Bangalore

26 vehicle categories (14 used in our subset)

Multiple camera viewpoints

High-quality annotations

📄 Project Members – Group 41

Member 1 – Vehicle detection & model training

Member 2 – Tracking pipeline

Member 3 – Backend & integration

Member 4 – Report & analytics

Member 5 – Presentation & documentation

📚 References

UVH-26 Dataset – IISc Bangalore

YOLOv5 by Ultralytics

DeepSORT: Simple Online and Realtime Tracking

✔️ Status

This phase of the project is complete with:

Successful training

Successful tracking

Working pipeline

Documentation ready

Next steps involve integrating post-processing analytics (speed estimation, vehicle counts, congestion metrics).