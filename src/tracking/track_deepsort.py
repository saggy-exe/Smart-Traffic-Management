# track_deepsort.py
# Usage (from CMD):
#   D:\anaconda3\envs\traffic\python.exe "C:\Users\Sagnic Ghosh\track_deepsort.py" ^
#       --weights "C:\Users\Sagnic Ghosh\yolov5\runs\train\uvh26_yolo_smoke\weights\best.pt" ^
#       --source "D:\videos\traffic_test.mp4" ^
#       --output "D:\videos\traffic_test_tracked.mp4"

import sys
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np

# ---- point directly to your yolov5 repo ----
YOLO_ROOT = Path(r"C:\Users\Sagnic Ghosh\yolov5")
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device

from deep_sort_realtime.deepsort_tracker import DeepSort

# Your UVH-26 class names
CLASS_NAMES = [
    'hatchback', 'sedan', 'suv', 'muv', 'bus', 'truck', 'three-wheeler',
    'two-wheeler', 'lcv', 'mini-bus', 'tempo-traveller', 'bicycle', 'van', 'other'
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="path to YOLOv5 .pt weights")
    ap.add_argument("--source", type=str, required=True, help="input video path")
    ap.add_argument("--output", type=str, required=True, help="output video path")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (square)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    return ap.parse_args()


def main(opt):
    device = select_device("")  # auto GPU/CPU

    # Load YOLO model (data file only used for some internals; names we override)
    model = DetectMultiBackend(
        opt.weights,
        device=device,
        dnn=False,
        data=YOLO_ROOT / "data" / "coco128.yaml"
    )
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)

    model.eval()

    # DeepSORT tracker
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
    )

    # Open video
    cap = cv2.VideoCapture(opt.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {opt.source}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(opt.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    print(f"[INFO] Starting tracking on {opt.source}")
    print(f"[INFO] Saving to {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ----- Preprocess -----
        # letterbox returns HWC (BGR)
        img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
        # BGR -> RGB, HWC -> CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # NCHW

        # ----- Inference -----
        with torch.no_grad():
            pred = model(img_tensor, augment=False)

        # ----- NMS -----
        dets = non_max_suppression(pred, opt.conf, opt.iou, classes=None, agnostic=False)[0]

        detections_for_tracker = []
        if dets is not None and len(dets):
            # Rescale boxes back to original frame
            dets[:, :4] = scale_boxes(img_tensor.shape[2:], dets[:, :4], frame.shape).round()

            for *xyxy, conf, cls in dets:
                x1, y1, x2, y2 = [float(x) for x in xyxy]
                w_box = x2 - x1
                h_box = y2 - y1
                if w_box <= 0 or h_box <= 0:
                    continue
                detections_for_tracker.append(([x1, y1, w_box, h_box], float(conf), int(cls)))

        # ----- Update tracker -----
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        # ----- Draw tracks -----
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cls = track.get_det_class()

            # basic clipping
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            y2 = max(0, min(y2, frame.shape[0] - 1))

            label = f"ID {tid}"
            if cls is not None and 0 <= int(cls) < len(CLASS_NAMES):
                label += f" {CLASS_NAMES[int(cls)]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed frame {frame_idx}")

    cap.release()
    writer.release()
    print("[INFO] Done. Output saved at:", out_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
