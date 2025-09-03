#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import json


DEFAULT_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

 


def load_face_classifier(cascade_path: Path) -> cv2.CascadeClassifier:
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        raise RuntimeError(
            f"Failed to load cascade classifier from {cascade_path}. "
            "The file may be corrupted."
        )
    return classifier


def detect_faces(
    frame_bgr: np.ndarray,
    classifier: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return list(faces)


def draw_face_boxes(
    frame_bgr: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, thickness)
    return frame_bgr


def is_video_source(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


def is_image_source(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))


def resolve_backend(backend_name: Optional[str]) -> Optional[int]:
    if not backend_name:
        return None
    name = backend_name.strip().upper()
    mapping = {
        "ANY": cv2.CAP_ANY,
        "AVFOUNDATION": cv2.CAP_AVFOUNDATION,
        "QT": cv2.CAP_QT if hasattr(cv2, "CAP_QT") else None,
        "V4L2": cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else None,
    }
    return mapping.get(name, None)


def run_on_webcam(
    classifier: cv2.CascadeClassifier,
    camera_index: int,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
    backend_name: Optional[str] = None,
    recognizer: Optional[any] = None,
    id_to_name: Optional[dict] = None,
    target_name_for_red: Optional[str] = None,
    recognition_threshold: float = 70.0,
) -> None:
    preferred_backend = resolve_backend(backend_name)
    backend_candidates = []
    if preferred_backend is not None:
        backend_candidates.append(preferred_backend)
    # Common backends on macOS
    backend_candidates.extend([
        x for x in [
            cv2.CAP_AVFOUNDATION,
            cv2.CAP_ANY,
            getattr(cv2, "CAP_QT", None),
        ] if isinstance(x, int)
    ])

    cap = None
    last_open_error = None
    for backend in backend_candidates:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                break
            if cap is not None:
                cap.release()
        except Exception as exc:  # noqa: BLE001
            last_open_error = exc
            if cap is not None:
                cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        msg = (
            f"Unable to open webcam at index {camera_index}. "
            "On macOS, ensure your terminal/IDE has Camera permission in System Settings → Privacy & Security → Camera."
        )
        if last_open_error:
            msg += f" Last error: {last_open_error}"
        raise RuntimeError(msg)

    print("Press 'q' or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam. Exiting.")
                break
            faces = detect_faces(frame, classifier, scale_factor, min_neighbors, min_size)
            annotated = frame
            if recognizer is not None and id_to_name is not None and len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.resize(roi_gray, (200, 200))
                    label_id, confidence = recognizer.predict(roi_gray)
                    name = id_to_name.get(str(label_id), "unknown")
                    box_color = (0, 255, 0)
                    if target_name_for_red and name == target_name_for_red and confidence <= recognition_threshold:
                        box_color = (0, 0, 255)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(annotated, f"{name}:{confidence:.0f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
            else:
                annotated = draw_face_boxes(frame, faces)
            cv2.imshow("Face Detection - Webcam", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_on_video(
    classifier: cv2.CascadeClassifier,
    video_path: Path,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    print("Press 'q' or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of video or failed to read frame.")
                break
            faces = detect_faces(frame, classifier, scale_factor, min_neighbors, min_size)
            annotated = draw_face_boxes(frame, faces)
            cv2.imshow(f"Face Detection - {video_path.name}", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_on_image(
    classifier: cv2.CascadeClassifier,
    image_path: Path,
    output_path: Path,
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
    recognizer: Optional[any] = None,
    id_to_name: Optional[dict] = None,
    target_name_for_red: Optional[str] = None,
    recognition_threshold: float = 70.0,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    faces = detect_faces(image, classifier, scale_factor, min_neighbors, min_size)
    annotated = image.copy()
    if recognizer is not None and id_to_name is not None and len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (200, 200))
            label_id, confidence = recognizer.predict(roi_gray)
            name = id_to_name.get(str(label_id), "unknown")
            box_color = (0, 255, 0)
            if target_name_for_red and name == target_name_for_red and confidence <= recognition_threshold:
                box_color = (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(annotated, f"{name}:{confidence:.0f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
    else:
        annotated = draw_face_boxes(image, faces)

    window_title = f"Face Detection - {image_path.name} ({len(faces)} faces)"
    cv2.imshow(window_title, annotated)
    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to {output_path}")
    print("Press any key to close window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple face detection using OpenCV Haar cascade")
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="'webcam' for live camera, a path to an image or video file, or a camera index (e.g., '0').",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to use when source is 'webcam' or an integer (default: 0).",
    )
    parser.add_argument(
        "--cascade",
        type=str,
        default=str(DEFAULT_CASCADE_PATH),
        help="Path to Haar cascade XML. Will be downloaded if missing.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Parameter specifying how much the image size is reduced at each image scale.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Parameter specifying how many neighbors each candidate rectangle should have to retain it.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        default=(30, 30),
        metavar=("W", "H"),
        help="Minimum possible face size (W H). Faces smaller than this are ignored.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="When running on an image, optional output path to save the annotated result.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="",
        help="Optional video backend (e.g., AVFOUNDATION, ANY).",
    )
    # Enrollment and training
    parser.add_argument("--enroll", action="store_true", help="Collect face samples for a given --name using webcam.")
    parser.add_argument("--name", type=str, default="", help="Person name for enrollment or recognition.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to capture during enrollment.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to store/read training images (organized as data/<name>/*.png).")
    parser.add_argument("--train", action="store_true", help="Train the recognizer from images in --data-dir.")
    parser.add_argument("--model", type=str, default=str(Path("models")/"lbph_model.xml"), help="Path to save/load the trained LBPH model.")
    parser.add_argument("--labels", type=str, default=str(Path("models")/"labels.json"), help="Path to save/load label mapping.")
    parser.add_argument("--recognize-name", type=str, default="", help="If set, draw this person's face in red when recognized.")
    parser.add_argument("--threshold", type=float, default=70.0, help="LBPH confidence threshold (lower is better).")
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_enrollment_samples(
    name: str,
    classifier: cv2.CascadeClassifier,
    num_samples: int,
    data_dir: Path,
    camera_index: int,
    backend_name: Optional[str],
    scale_factor: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
) -> None:
    if not name:
        raise ValueError("--name is required for enrollment")
    person_dir = data_dir / name
    ensure_dir(person_dir)

    preferred_backend = resolve_backend(backend_name)
    backend = preferred_backend if preferred_backend is not None else cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam for enrollment. Check camera permissions.")

    print(f"Enrolling '{name}'. Look at the camera. Press 'q' to stop early.")
    captured = 0
    try:
        while captured < num_samples:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam. Exiting.")
                break
            faces = detect_faces(frame, classifier, scale_factor, min_neighbors, min_size)
            for (x, y, w, h) in faces:
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (200, 200))
                out_path = person_dir / f"sample_{int(cv2.getTickCount())}.png"
                cv2.imwrite(str(out_path), roi_gray)
                captured += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
                break
            cv2.imshow("Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if captured % 10 == 0 and captured > 0:
                print(f"Captured {captured}/{num_samples} samples...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Enrollment complete. Captured {captured} samples in {person_dir}")


def load_images_and_labels(data_dir: Path) -> tuple[list[np.ndarray], list[int], dict]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    name_to_id: dict[str, int] = {}
    next_id = 0
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        name = person_dir.name
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1
        label_id = name_to_id[name]
        for img_path in sorted(person_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(label_id)
    return images, labels, {v: k for k, v in name_to_id.items()}


def train_lbph(data_dir: Path, model_path: Path, labels_path: Path) -> None:
    images, labels, id_to_name = load_images_and_labels(data_dir)
    if not images:
        raise RuntimeError(f"No training images found in {data_dir}. Enroll first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    ensure_dir(model_path.parent)
    recognizer.write(str(model_path))
    ensure_dir(labels_path.parent)
    labels_path.write_text(json.dumps({str(k): v for k, v in id_to_name.items()}, indent=2))
    print(f"Saved model to {model_path}\nSaved labels to {labels_path}")


def load_recognizer(model_path: Path, labels_path: Path) -> tuple[any, dict]:
    if not model_path.exists() or not labels_path.exists():
        raise RuntimeError("Model or labels not found. Train first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))
    id_to_name = json.loads(labels_path.read_text())
    return recognizer, id_to_name


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Normalize source
    source_str = str(args.source).strip()
    cascade_path = Path(args.cascade).expanduser().resolve()
    classifier = load_face_classifier(cascade_path)

    # Enrollment
    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.enroll:
        collect_enrollment_samples(
            name=args.name,
            classifier=classifier,
            num_samples=args.num_samples,
            data_dir=data_dir,
            camera_index=args.camera_index,
            backend_name=args.backend,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
        )
        return 0

    # Training
    model_path = Path(args.model).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    if args.train:
        train_lbph(data_dir=data_dir, model_path=model_path, labels_path=labels_path)
        return 0

    # Recognition config (optional)
    recognizer = None
    id_to_name = None
    if args.recognize_name:
        try:
            recognizer, id_to_name = load_recognizer(model_path, labels_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: recognition disabled: {exc}")

    # If source is a digit, treat as camera index
    if source_str.lower() == "webcam" or source_str.isdigit():
        camera_index = args.camera_index if not source_str.isdigit() else int(source_str)
        run_on_webcam(
            classifier=classifier,
            camera_index=camera_index,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
            backend_name=args.backend,
            recognizer=recognizer,
            id_to_name=id_to_name,
            target_name_for_red=args.recognize_name if args.recognize_name else None,
            recognition_threshold=float(args.threshold),
        )
        return 0

    source_path = Path(source_str).expanduser().resolve()
    if not source_path.exists():
        print(f"Source not found: {source_path}", file=sys.stderr)
        return 2

    if is_image_source(source_path.name):
        output_path = Path(args.output).expanduser().resolve() if args.output else None
        run_on_image(
            classifier=classifier,
            image_path=source_path,
            output_path=output_path if output_path else None,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
            recognizer=recognizer,
            id_to_name=id_to_name,
            target_name_for_red=args.recognize_name if args.recognize_name else None,
            recognition_threshold=float(args.threshold),
        )
    elif is_video_source(source_path.name):
        run_on_video(
            classifier=classifier,
            video_path=source_path,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
        )
    else:
        print(
            "Unrecognized source type. Use 'webcam', a camera index, an image file, or a video file.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
