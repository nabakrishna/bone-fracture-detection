#!/usr/bin/env python3

#epoch is basicaly from 50 to 100 and batch size is from 16 to 8 and confidence threshold is from 0.5 to 0.3 and learning rate is from 0.01 to 0.001
""" 
Bone Fracture Detection System using YOLOv8
============================================
A complete pipeline for training and running inference
on bone fracture X-ray images.

Usage:
    # Train
    python bone_fracture.py train --data ./my_data --epochs 100

    # Detect single image
    python bone_fracture.py detect --image xray.jpg

    # Detect batch
    python bone_fracture.py detect --images-dir ./xrays/

    # Check status
    python bone_fracture.py status
"""
#importing the all the req libary
import os
import sys
import yaml
import shutil
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

__all__ = ["BoneFractureDetectionSystem"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def install_requirements() -> None:
    """Install required packages if missing."""
    try:
        import ultralytics  # noqa: F401
        logger.info("Required packages already installed.")
    except ImportError:
        packages = [
            "ultralytics>=8.0.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "pyyaml>=6.0",
            "scikit-learn>=1.3.0",
        ]
        logger.info("Installing required packages …")
        for pkg in packages:
            os.system(f"{sys.executable} -m pip install -q {pkg}")
        logger.info("All packages installed.")


def detect_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
            return "cuda"
    except Exception:
        pass
    logger.info("No GPU detected — using CPU.")
    return "cpu"


def is_valid_image(path: Path) -> bool:
    """Return True if the file can be decoded by OpenCV."""
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        return False


def validate_yolo_label(label_path: Path, num_classes: int) -> List[str]:
    """
    Validate a YOLO-format label file.
    Returns a list of error strings (empty == valid).
    """
    errors: List[str] = []
    try:
        with open(label_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(
                        f"{label_path.name}:{line_no} — expected 5 values, got {len(parts)}"
                    )
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path.name}:{line_no} — non-numeric value")
                    continue
                if cls_id < 0 or cls_id >= num_classes:
                    errors.append(
                        f"{label_path.name}:{line_no} — class {cls_id} out of range [0, {num_classes})"
                    )
                for v in coords:
                    if v < 0.0 or v > 1.0:
                        errors.append(
                            f"{label_path.name}:{line_no} — coordinate {v} outside [0, 1]"
                        )
                        break
    except Exception as exc:
        errors.append(f"{label_path.name} — could not read: {exc}")
    return errors


def collect_image_files(directory: Path) -> List[Path]:
    """Collect all supported image files from *directory*."""
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    # deduplicate (upper/lower may overlap on case-insensitive FS)
    return sorted(set(files))


# ---------------------------------------------------------------------------
# main system--class definition and methodsf
# ---------------------------------------------------------------------------
class BoneFractureDetectionSystem:
    """End-to-end bone fracture detection: data prep → training → inference."""

    # Default class catalogue (override via config or constructor)
    DEFAULT_CLASSES = [
        "Bone Abnormality",
        "Bone Injury",
        "Foreign Object",
        "Bone Fracture",
        "Metal",
        "Periosteal Reaction",
        "Pronator Sign",
        "Soft Tissue",
        "Text",
    ]

    def __init__(
        self,
        project_root: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        model_variant: str = "yolov8s",
    ) -> None:
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._setup_directories()

        self.config: Dict[str, Any] = {
            # training
            "epochs": 50,
            "batch_size": 8,
            "img_size": 640,
            "learning_rate": 0.01,
            "patience": 20,
            # model
            "model_name": model_variant,
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            # data split
            "train_ratio": 0.70,
            "val_ratio": 0.20,
            "test_ratio": 0.10,
            # classes
            "class_names": class_names or self.DEFAULT_CLASSES,
            # device — auto-detected
            "device": detect_device(),
        }

        self.model = None
        self.trained_model_path: Optional[str] = None

        logger.info(f"System initialised — root: {self.project_root}")

    # ------------------------------------------------------------------ dirs
    def _setup_directories(self) -> None:
        self.dirs = {
            "data":        self.project_root / "data",
            "raw_images":  self.project_root / "data" / "raw" / "images",
            "raw_labels":  self.project_root / "data" / "raw" / "labels",
            "processed":   self.project_root / "data" / "processed",
            "models":      self.project_root / "models",
            "pretrained":  self.project_root / "models" / "pretrained",
            "trained":     self.project_root / "models" / "trained",
            "results":     self.project_root / "results",
            "predictions": self.project_root / "results" / "predictions",
            "metrics":     self.project_root / "results" / "metrics",
        }
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- data prep
    def _create_data_yaml(self) -> Path:
        """Write ``data.yaml`` consumed by YOLO training."""
        content = {
            "path": str(self.dirs["processed"]),
            "train": "train/images",
            "val":   "val/images",
            "test":  "test/images",
            "nc":    len(self.config["class_names"]),
            "names": self.config["class_names"],      # list, not dict
        }
        yaml_path = self.dirs["processed"] / "data.yaml"
        with open(yaml_path, "w") as fh:
            yaml.dump(content, fh, default_flow_style=False)
        logger.info(f"Created {yaml_path}")
        return yaml_path
    

    #simple function to clean the processed directory before creating new splits, to avoid stale data on re-run
    def _clean_processed(self) -> None:
        """Remove old processed splits to avoid stale data on re-run."""
        for split in ("train", "val", "test"):
            split_dir = self.dirs["processed"] / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        logger.info("Cleaned previous processed data.")

    #audit raw data for validity and consistency, returning valid image paths and counts of issues found
    def _audit_raw_data(self) -> tuple[List[Path], int, int]:
        """
        Scan raw images/labels, validate them, return
        ``(valid_image_paths, n_corrupt_images, n_label_errors)``.
        """
        image_files = collect_image_files(self.dirs["raw_images"])
        if not image_files:
            raise FileNotFoundError(
                f"No images found in {self.dirs['raw_images']}.\n"
                f"  Place .jpg/.png images there and matching .txt labels in {self.dirs['raw_labels']}."
            )

        valid: List[Path] = []
        n_corrupt = 0
        n_label_errors = 0
        num_classes = len(self.config["class_names"])

        for img in image_files:
            # check image readability
            if not is_valid_image(img):
                logger.warning(f"Corrupt/unreadable image skipped: {img.name}")
                n_corrupt += 1
                continue

            # check label (optional — images without labels are "negatives") imp for a good training 
            label = self.dirs["raw_labels"] / f"{img.stem}.txt"
            if label.exists():
                errs = validate_yolo_label(label, num_classes)
                if errs:
                    for e in errs:
                        logger.warning(f"Label error: {e}")
                    n_label_errors += len(errs)
                    continue  # skip this image entirely

            valid.append(img)

        logger.info(
            f"Audit: {len(valid)} valid, {n_corrupt} corrupt, "
            f"{n_label_errors} label errors out of {len(image_files)} images."
        )
        return valid, n_corrupt, n_label_errors

    def split_dataset(self) -> bool:
        """Split raw images+labels into train / val / test."""
        try:
            valid_images, n_corrupt, n_label_err = self._audit_raw_data()
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return False

        if len(valid_images) < 3:
            logger.error("Need at least 3 valid images to create a split.")
            return False

        self._clean_processed()

        # stratified split is not possible (no per-image class), use random
        train_files, temp_files = train_test_split(
            valid_images,
            train_size=self.config["train_ratio"],
            random_state=42,
        )

        val_ratio_adjusted = self.config["val_ratio"] / (
            self.config["val_ratio"] + self.config["test_ratio"]
        )
        if len(temp_files) < 2:
            val_files, test_files = temp_files, []
        else:
            val_files, test_files = train_test_split(
                temp_files,
                train_size=val_ratio_adjusted,
                random_state=42,
            )

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in splits.items():
            img_dir = self.dirs["processed"] / split_name / "images"
            lbl_dir = self.dirs["processed"] / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            n_img = n_lbl = 0
            for img_file in files:
                shutil.copy2(img_file, img_dir / img_file.name)
                n_img += 1
                label_file = self.dirs["raw_labels"] / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, lbl_dir / label_file.name)
                    n_lbl += 1

            logger.info(f"  {split_name:>5}: {n_img} images, {n_lbl} labels")

        logger.info("Dataset split complete.")
        return True

    def validate_dataset(self) -> bool:
        """Quick structural check on the processed directory."""
        required = ["train/images", "train/labels", "val/images", "val/labels"]
        for rel in required:
            d = self.dirs["processed"] / rel
            if not d.exists() or not any(d.iterdir()):
                logger.error(f"Missing or empty: {d}")
                return False

        yaml_path = self.dirs["processed"] / "data.yaml"
        if not yaml_path.exists():
            logger.error("data.yaml missing.")
            return False

        logger.info("Dataset validation passed.")
        return True

    def prepare_dataset(self) -> bool:
        """Full data-prep pipeline: audit → split → yaml → validate."""
        logger.info("Preparing dataset …")
        if not self.split_dataset():
            return False
        self._create_data_yaml()
        return self.validate_dataset()

    # --------------------------------------------------------------- training-------------------------------
    def train_model(self, **kwargs) -> Optional[str]:
        """
        Train a YOLOv8 model on the prepared dataset.

        Returns the path to ``best.pt`` on success, ``None`` on failure.
        """
        from ultralytics import YOLO
#in this we fine tune the yolo model yolov8 but for upgration and more stable we can go for yolo11l or yolo26l(latest model)
        yaml_path = self.dirs["processed"] / "data.yaml"
        if not yaml_path.exists():
            logger.error("data.yaml not found. Run prepare_dataset() first.")
            return None

        logger.info(f"Loading pretrained {self.config['model_name']} …")
        self.model = YOLO(f"{self.config['model_name']}.pt")
#imp may some more parameter may be addfor accuracy in specific class 
        params = {
            "data":        str(yaml_path),
            "epochs":      kwargs.get("epochs",  self.config["epochs"]),
            "batch":       kwargs.get("batch",   self.config["batch_size"]),
            "imgsz":       kwargs.get("imgsz",   self.config["img_size"]),
            "lr0":         kwargs.get("lr0",     self.config["learning_rate"]),
            "patience":    kwargs.get("patience", self.config["patience"]),
            "device":      kwargs.get("device",  self.config["device"]),
            "project":     str(self.dirs["trained"]),
            "name":        "bone_fracture_detector",
            "save_period":  10,
            "save":         True,
            "plots":        True,
            "exist_ok":     True,
            "verbose":      True,
        }

        logger.info("Training parameters:")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")

        t0 = time.time()
        try:
            results = self.model.train(**params)
        except Exception as exc:
            logger.exception(f"Training failed: {exc}")
            logger.info(
                "Hints: ensure ≥10 images, correct label format, "
                "reduce batch_size if OOM."
            )
            return None

        elapsed = time.time() - t0
        logger.info(f"Training finished in {elapsed / 60:.1f} min.")

        best_pt = (
            self.dirs["trained"]
            / "bone_fracture_detector"
            / "weights"
            / "best.pt" 
        )
        if not best_pt.exists():
            logger.error(f"best.pt not found at {best_pt}")
            return None

        self.trained_model_path = str(best_pt)
        self.model = YOLO(self.trained_model_path)
        logger.info(f"Custom model ready: {best_pt}")

        self._save_training_info(results, elapsed, best_pt)
        return self.trained_model_path

    def _save_training_info(
        self, results: Any, elapsed: float, model_path: Path
    ) -> None:
        info_path = self.dirs["metrics"] / "training_info.yaml"

        # extract metrics from results safely
        metrics: Dict[str, Any] = {}
        try:
            box = results.results_dict
            metrics = {k: float(v) for k, v in box.items()}
        except Exception:
            pass

        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_minutes": round(elapsed / 60, 2),
            "model_path": str(model_path),
            "config": self.config,
            "metrics": metrics,
        }
        with open(info_path, "w") as fh:
            yaml.dump(payload, fh, default_flow_style=False)
        logger.info(f"Training info saved → {info_path}")

    def evaluate_model(self, model_path: Optional[str] = None) -> Dict:
        """Run evaluation on the test (or val) set and return metrics."""
        from ultralytics import YOLO

        if not self.load_model(model_path):
            return {}

        yaml_path = self.dirs["processed"] / "data.yaml"
        if not yaml_path.exists():
            logger.error("data.yaml not found.")
            return {}

        # Prefer test set; fall back to val it 
        test_imgs = self.dirs["processed"] / "test" / "images"
        split = "test" if test_imgs.exists() and any(test_imgs.iterdir()) else "val"
        logger.info(f"Evaluating on '{split}' split …")

        try:
            metrics = self.model.val(
                data=str(yaml_path),
                split=split,
                device=self.config["device"],
                verbose=False,
            )
            result = {
                "mAP50":    float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall":   float(metrics.box.mr),
            }
            logger.info(f"Evaluation results: {result}")
            return result
        except Exception as exc:
            logger.exception(f"Evaluation failed: {exc}")
            return {}

    # --------------------------------------------------------------- loading
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load a YOLO model.  Priority: arg → stored → on-disk → pretrained."""
        from ultralytics import YOLO

        candidates = [
            model_path,
            self.trained_model_path,
            str(
                self.dirs["trained"]
                / "bone_fracture_detector"
                / "weights"
                / "best.pt" 
            ),
        ]

        for path in candidates:
            if path and Path(path).is_file():
                self.model = YOLO(path)
                self.trained_model_path = path
                logger.info(f"Loaded model: {path}")
                return True

        # Fallback to pretrained (not fracture-specialised)
        logger.warning(
            f"No trained model found — falling back to pretrained "
            f"{self.config['model_name']} (run train_model() first)."
        )
        self.model = YOLO(f"{self.config['model_name']}.pt")
        return True

    # ------------------------------------------------------------- inference
    def detect_fractures(
        self,
        image_path: str,
        model_path: Optional[str] = None,
        save_results: bool = True,
        show: bool = False,
    ) -> Dict[str, Any]:
        """
        Run detection on a single image.

        Returns a dict with keys:
            image_path, boxes, confidences, classes,
            class_names, num_detections, output_path (if saved).
        """
        if not self.model:
            self.load_model(model_path)

        img_p = Path(image_path)
        if not img_p.is_file():
            logger.error(f"Image not found: {img_p}")
            return {}

        if not is_valid_image(img_p):
            logger.error(f"Image unreadable: {img_p}")
            return {}

        logger.info(f"Analysing {img_p.name} …")

        results = self.model(
            str(img_p),
            conf=self.config["confidence_threshold"],
            iou=self.config["iou_threshold"],
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes

        detections: Dict[str, Any] = {
            "image_path": str(img_p),
            "boxes": boxes.xyxy.cpu().numpy()   if boxes and len(boxes) else np.empty((0, 4)),
            "confidences": boxes.conf.cpu().numpy() if boxes and len(boxes) else np.empty(0),
            "classes": boxes.cls.cpu().numpy()   if boxes and len(boxes) else np.empty(0),
            "class_names": (
                [r.names[int(c)] for c in boxes.cls] if boxes and len(boxes) else []
            ),
            "num_detections": int(len(boxes) if boxes else 0),
        }

        n = detections["num_detections"]
        if n:
            avg = float(np.mean(detections["confidences"]))
            # Group by class for summary
            from collections import Counter
            counts = Counter(detections["class_names"])
            summary = ", ".join(f"{v}× {k}" for k, v in counts.items())
            logger.info(f"  {n} detection(s) — {summary}  (avg conf {avg:.2f})")
        else:
            logger.info("  No detections.")

        if save_results and n:
            out = self.dirs["predictions"] / f"detected_{img_p.name}"
            annotated = r.plot()
            cv2.imwrite(str(out), annotated)
            detections["output_path"] = str(out)
            logger.info(f"  Saved → {out}")

        if show:
            self.visualize_results(detections, img_p)

        return detections

    def batch_detect(
        self,
        images_path: str,
        model_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Detect on every image in a directory."""
        d = Path(images_path)
        if not d.is_dir():
            logger.error(f"Not a directory: {d}")
            return []

        files = collect_image_files(d)
        if not files:
            logger.error(f"No images in {d}")
            return []

        logger.info(f"Batch: {len(files)} images")
        all_results: List[Dict[str, Any]] = []

        for idx, f in enumerate(files, 1):
            logger.info(f"[{idx}/{len(files)}] {f.name}")
            det = self.detect_fractures(str(f), model_path)
            all_results.append(det)

        # Summary
        total = sum(r.get("num_detections", 0) for r in all_results)
        with_det = sum(1 for r in all_results if r.get("num_detections", 0) > 0)
        logger.info("=" * 50)
        logger.info("BATCH SUMMARY")
        logger.info(f"  Images processed:       {len(all_results)}")
        logger.info(f"  Images with detections: {with_det}")
        logger.info(f"  Total detections:       {total}")
        logger.info(f"  Avg per image:          {total / max(len(all_results), 1):.2f}")
        return all_results

    # --------------------------------------------------------- visualisation
    def visualize_results(
        self,
        detection: Dict[str, Any],
        fallback_image: Optional[Path] = None,
    ) -> None:
        """Show detection result with matplotlib."""
        img_path = detection.get("output_path") or (
            str(fallback_image) if fallback_image else detection.get("image_path")
        )
        if not img_path or not Path(img_path).is_file():
            logger.warning("Nothing to visualise.")
            return

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        n = detection.get("num_detections", 0)

        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Detections: {n}")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------- full pipeline
    def create_and_train_model(self, **kwargs) -> bool:
        """End-to-end: prepare → train → evaluate."""
        logger.info("=" * 60)
        logger.info("FULL PIPELINE — prepare → train → evaluate")
        logger.info("=" * 60)

        if not self.prepare_dataset():
            return False

        path = self.train_model(**kwargs)
        if not path:
            return False

        logger.info("Running evaluation on test set …")
        self.evaluate_model()

        logger.info("=" * 60)
        logger.info(f"Pipeline complete.  Model → {path}")
        logger.info("=" * 60)
        return True

    # ------------------------------------------------------------- status
    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "dataset_prepared": False,
            "model_trained": False,
            "ready_for_inference": False,
            "trained_model_path": None,
            "device": self.config["device"],
        }
        if (self.dirs["processed"] / "data.yaml").exists():
            status["dataset_prepared"] = True
        best = (
            self.dirs["trained"]
            / "bone_fracture_detector"
            / "weights"
            / "best.pt"
        )
        if best.is_file():
            status["model_trained"] = True
            status["trained_model_path"] = str(best)
            status["ready_for_inference"] = True
        return status

    def print_status(self) -> None:
        s = self.get_status()
        check = lambda b: "✅" if b else "❌"  # noqa: E731
        logger.info("SYSTEM STATUS")
        logger.info(f"  Device:             {s['device']}")
        logger.info(f"  Dataset prepared:   {check(s['dataset_prepared'])}")
        logger.info(f"  Model trained:      {check(s['model_trained'])}")
        logger.info(f"  Ready for inference:{check(s['ready_for_inference'])}")
        if s["trained_model_path"]:
            logger.info(f"  Model path:         {s['trained_model_path']}")
        if not s["dataset_prepared"]:
            logger.info(
                f"  → Place images in {self.dirs['raw_images']} "
                f"and labels in {self.dirs['raw_labels']}, then run prepare_dataset()."
            )
        elif not s["model_trained"]:
            logger.info("  → Run train_model().")
        else:
            logger.info("  → Use detect_fractures('image.jpg').")


# ---------------------------------------------------------------------------
# cli interface for efficient testing and usage
# ---------------------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bone Fracture Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Print system status")

    # prepare
    prep = sub.add_parser("prepare", help="Prepare dataset (split + yaml)")
    prep.add_argument("--root", default=".", help="Project root directory")

    # train
    tr = sub.add_parser("train", help="Train the model")
    tr.add_argument("--root",    default=".")
    tr.add_argument("--epochs",  type=int, default=50)
    tr.add_argument("--batch",   type=int, default=8)
    tr.add_argument("--imgsz",   type=int, default=640)
    tr.add_argument("--lr0",     type=float, default=0.01)
    tr.add_argument("--patience", type=int, default=20)
    tr.add_argument("--full", action="store_true",
                    help="Run full pipeline (prepare + train + evaluate)")

    # detect
    det = sub.add_parser("detect", help="Run inference")
    det.add_argument("--root",       default=".")
    det.add_argument("--image",      help="Single image path")
    det.add_argument("--images-dir", help="Directory of images")
    det.add_argument("--model",      help="Path to .pt model file")
    det.add_argument("--conf",       type=float, default=0.5)
    det.add_argument("--show",       action="store_true")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate on test/val set")
    ev.add_argument("--root",  default=".")
    ev.add_argument("--model", help="Path to .pt model file")

    return p


def main() -> None:
    install_requirements()
    parser = build_cli()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    root = getattr(args, "root", ".")
    system = BoneFractureDetectionSystem(project_root=root)

    if args.command == "status":
        system.print_status()

    elif args.command == "prepare":
        system.prepare_dataset()

    elif args.command == "train":
        kw = {k: getattr(args, k) for k in ("epochs", "batch", "imgsz", "lr0", "patience")}
        if args.full:
            system.create_and_train_model(**kw)
        else:
            system.train_model(**kw)

    elif args.command == "detect":
        system.config["confidence_threshold"] = args.conf
        if args.image:
            system.detect_fractures(args.image, model_path=args.model, show=args.show)
        elif args.images_dir:
            system.batch_detect(args.images_dir, model_path=args.model)
        else:
            logger.error("Provide --image or --images-dir.")

    elif args.command == "evaluate":
        system.evaluate_model(model_path=args.model)


if __name__ == "__main__":
    main()
