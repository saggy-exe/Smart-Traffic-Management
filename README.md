# 🚦 **Smart Traffic Management System (Group 41)**

A real-time **vehicle detection**, **classification**, and **tracking system** using:

* **YOLOv5** (custom-trained on UVH-26 dataset)

* **DeepSORT** (multi-object tracking)

The system processes traffic videos and outputs tracks with unique IDs for monitoring and analytics.


## 📌 Features

* Detects vehicles across **14 UVH-26 classes**

* Tracks vehicles with **consistent IDs**

* Works on real traffic videos

* Generates analytics (movement patterns, vehicle count, etc.)


## 🧠 Tech Stack

* Python 3.10

* YOLOv5 (Ultralytics)

* DeepSORT

* CUDA GPU Acceleration

* UVH-26 Dataset


## 📦 Installation
git clone https://github.com/<your-username>/SmartTrafficSystem.git
cd SmartTrafficSystem

conda create -n traffic python=3.10
conda activate traffic

pip install -r requirements.txt
pip install deep-sort-realtime


Place your trained YOLO model in:

SmartTrafficSystem/models/best.pt

## ▶️ Run Detection
python yolov5/detect.py --weights models/best.pt --source sample.jpg

## ▶️ Run Tracking (YOLO + DeepSORT)
python src/track_deepsort.py \
&nbsp;&nbsp;--weights models/best.pt \
&nbsp;&nbsp;--source input_video.mp4 \
&nbsp;&nbsp;--output outputs/tracked_video.mp4

## 📁 Project Structure
SmartTrafficSystem/

│

├── src/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Custom scripts (tracking, dataset tools, etc.)

├── models/ &nbsp;&nbsp;&nbsp;&nbsp; # best.pt (ignored in repository)

├── outputs/ &nbsp;&nbsp;&nbsp;&nbsp; # Result images/videos

├── docs/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # PDF + Presentation

├── yolov5/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # YOLOv5 framework

│

└── README.md

## 📊 Dataset

**UVH-26 Dataset (IISc Bangalore)**
* Converted into YOLO format with 14 vehicle categories:

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; hatchback, sedan, suv, muv, bus, truck, three-wheeler, two-wheeler, lcv, mini-bus, tempo-traveller, bicycle, van, other.

