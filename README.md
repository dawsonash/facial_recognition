# Face Detection with OpenCV

A minimal Python project that detects human faces using OpenCV's Haar cascade and outlines them on webcam, video, or image input.

## Setup

1. Create a virtual environment and install dependencies:

```bash
cd "/Users/dawsonash/Documents/AI projects/Facial_Recognition_project"
python3 -m venv .venv
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/pip" install -r requirements.txt
```

## Usage

- Webcam (default camera index 0):

```bash
".venv/bin/python" face_detect.py --source webcam
```

- Specific webcam index (e.g., 1):

```bash
".venv/bin/python" face_detect.py --source 1 --camera-index 1
```

- Image file:

```bash
".venv/bin/python" face_detect.py --source path/to/image.jpg --output annotated.jpg
```

- Video file:

```bash
".venv/bin/python" face_detect.py --source path/to/video.mp4
```

- Adjustable parameters (examples):

```bash
".venv/bin/python" face_detect.py --source webcam \
  --scale-factor 1.1 --min-neighbors 5 --min-size 30 30
```

## Face Enrollment and Recognition (LBPH)

Install contrib build (already in requirements): `opencv-contrib-python`.

1) Enroll your face (captures samples from webcam):
```bash
".venv/bin/python" face_detect.py --enroll --name "YourName" --num-samples 120 --backend AVFOUNDATION
```
Images are saved under `data/YourName/*.png`.

2) Train the recognizer:
```bash
".venv/bin/python" face_detect.py --train --data-dir data --model models/lbph_model.xml --labels models/labels.json
```

3) Run recognition (your face outlined in red when recognized):
```bash
".venv/bin/python" face_detect.py --source webcam --recognize-name "YourName" --threshold 70 --backend AVFOUNDATION
```
Lower `--threshold` makes recognition stricter; higher makes it more permissive.

Notes:
- The Haar cascade is loaded from OpenCV's bundled data path, so no network download is needed.
- Press `q` or `ESC` to quit live windows.

## macOS Camera Permissions
If the webcam window is black or access is denied, grant camera permission to your terminal or IDE:
- System Settings → Privacy & Security → Camera → enable for your app.
