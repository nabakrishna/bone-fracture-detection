#new test file for detection

#!/usr/bin/env python3
"""
Bone Fracture Detection - Clean Summary Output
===============================================
Focused on detecting bone fractures with clean, medical-report style output.
Automatically ignores 'text' detections and focuses on fractures.

Usage:
    python detect_fractures.py --model best.pt --image xray.jpg
    python detect_fractures.py --model best.pt --image xray.jpg --conf 0.3 --save result.jpg
"""

import argparse
import sys
from pathlib import Path

from matplotlib.pyplot import show

try:
    from ultralytics import YOLO
    import cv2
    import torch
except ImportError:
    print("Installing required packages...")
    import os
    os.system(f"{sys.executable} -m pip install -q ultralytics opencv-python torch")
    from ultralytics import YOLO
    import cv2
    import torch


def detect_fractures(
    model_path: str,
    image_path: str,
    conf: float = 0.3,
    iou: float = 0.45,
    save_path: str = None,
    show: bool = False,
    verbose: bool = False
):
    """
    Detect bone fractures in an X-ray image.
    
    Args:
        model_path: Path to trained model (.pt file)
        image_path: Path to X-ray image
        conf: Confidence threshold (default 0.3 for medical imaging)
        iou: IoU threshold for NMS
        save_path: Optional path to save annotated image
        show: Display the result
        verbose: Show all detections, not just fractures
    """
    model_path = Path(r"C:\Users\N. K. Hazarika\Downloads\best (3).pt")
    # #"C:\Users\N. K. Hazarika\Downloads\best (3).pt"
    image_path = Path(r"D:\bone-fracture-detection\data\raw\images\0919_1128064861_06_WRI-R2_M004.png")
    # image_path = Path(r"D:\bone-fracture-detection\testing img\Screenshot 2026-02-12 175939.png")
    #D:\bone-fracture-detection\testing img\Screenshot 2026-02-12 175939.png
    # #D:\bone-fracture-detection\data\raw\images\0919_1128064861_06_WRI-R2_M004.png
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return None
    
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        return None
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = YOLO(str(model_path))
    
    # Define class categories
    FRACTURE_CLASSES = ['fracture', 'bone fracture', 'bone injury', 'bonelesion']
    IGNORE_CLASSES = ['text']
    
    print(f"\n{'='*70}")
    print("BONE FRACTURE DETECTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Image:      {image_path.name}")
    print(f"Model:      {model_path.name}")
    print(f"Device:     {device.upper()}")
    print(f"Confidence: {conf}")
    print(f"{'='*70}\n")
    
    # Run inference
    results = model(
        str(image_path),
        conf=conf,
        iou=iou,
        device=device,
        verbose=False
    )
    
    result = results[0]
    boxes = result.boxes
    
    if boxes is None or len(boxes) == 0:
        print("❌ NO DETECTIONS FOUND")
        print("\nRecommendations:")
        print("  • Lower confidence threshold (try --conf 0.2)")
        print("  • Check image quality and clarity")
        print("  • Verify X-ray is properly positioned")
        return None
    
    # Categorize detections
    fracture_detections = []
    other_detections = []
    ignored_count = 0
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        class_name = result.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        det_info = {
            'class': class_name,
            'confidence': conf_score,
            'box': (int(x1), int(y1), int(x2), int(y2)),
            'center': (int((x1+x2)/2), int((y1+y2)/2)),
            'width': int(x2 - x1),
            'height': int(y2 - y1)
        }
        
        if class_name.lower() in IGNORE_CLASSES:
            ignored_count += 1
        elif class_name.lower() in FRACTURE_CLASSES:
            fracture_detections.append(det_info)
        else:
            other_detections.append(det_info)
    
    # MAIN RESULTS - FRACTURES ONLY
    print("┌" + "─"*68 + "┐")
    print("│" + " "*20 + "🔍 FRACTURE ANALYSIS" + " "*27 + "│")
    print("└" + "─"*68 + "┘\n")
    
    if fracture_detections:
        print(f"✅ FRACTURES DETECTED: {len(fracture_detections)}\n")
        
        # Sort by confidence
        fracture_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        for idx, det in enumerate(fracture_detections, 1):
            print(f"┌─ FRACTURE #{idx} " + "─"*54)
            print(f"│ Type:       {det['class']}")
            print(f"│ Confidence: {det['confidence']:.1%} ({'High' if det['confidence'] > 0.7 else 'Medium' if det['confidence'] > 0.5 else 'Low'})")
            x1, y1, x2, y2 = det['box']
            cx, cy = det['center']
            print(f"│ Location:   Top-Left ({x1}, {y1}), Bottom-Right ({x2}, {y2})")
            print(f"│ Center:     ({cx}, {cy})")
            print(f"│ Size:       {det['width']}×{det['height']} pixels")
            print(f"└" + "─"*68)
            print()
        
        # Statistics
        avg_conf = sum(d['confidence'] for d in fracture_detections) / len(fracture_detections)
        max_conf = max(d['confidence'] for d in fracture_detections)
        min_conf = min(d['confidence'] for d in fracture_detections)
        
        print("📊 STATISTICS:")
        print(f"   Average Confidence: {avg_conf:.1%}")
        print(f"   Highest Confidence: {max_conf:.1%}")
        print(f"   Lowest Confidence:  {min_conf:.1%}")
        
    else:
        print("✅ NO FRACTURES DETECTED")
        print("\nThis X-ray appears to show no visible fractures.")
        print("Note: Low confidence threshold ({conf}) was used.")
    
    # Additional findings (if any)
    print(f"\n{'─'*70}")
    if other_detections:
        print(f"\nℹ️  OTHER FINDINGS: {len(other_detections)}")
        for det in other_detections:
            print(f"   • {det['class']}: {det['confidence']:.1%} confidence")
    
    if ignored_count > 0:
        print(f"\n🚫 Ignored: {ignored_count} text annotation(s)")
    
    # Verbose output
    if verbose:
        print(f"\n{'─'*70}")
        print(f"VERBOSE: All {len(boxes)} raw detections:")
        for idx, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            print(f"  {idx}. {result.names[cls_id]}: {float(box.conf[0]):.3f}")
    
    print(f"\n{'='*70}\n")
    
    # Save annotated image (with only fracture boxes)
    if save_path:
        img = cv2.imread(str(image_path))
        
        for det in fracture_detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            
            # Color based on confidence: green (high), yellow (med), red (low)
            if conf > 0.7:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"FRACTURE {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 0), 2)
        
        cv2.imwrite(save_path, img)
        print(f"💾 Annotated image saved to: {save_path}\n")
    
    # Display
    if show:
        img = cv2.imread(str(image_path))

        # cv2.namedWindow('Fracture Detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Fracture Detection', 900, 700)

        for det in fracture_detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 165, 255)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"FRACTURE {conf:.0%}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2)
        cv2.namedWindow('Fracture Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Fracture Detection', 600, 800)
        
        cv2.imshow('Fracture Detection', img)
        print("👁️  Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return {
        'fracture_count': len(fracture_detections),
        'fractures': fracture_detections,
        'other_findings': other_detections,
        'total_detections': len(boxes),
        'ignored_count': ignored_count
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect bone fractures in X-ray images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python detect_fractures.py --model best.pt --image xray.jpg
  
  # Lower confidence for sensitive detection
  python detect_fractures.py --model best.pt --image xray.jpg --conf 0.2
  
  # Save annotated result
  python detect_fractures.py --model best.pt --image xray.jpg --save result.jpg
  
  # Show all detections (verbose)
  python detect_fractures.py --model best.pt --image xray.jpg --verbose
        """
    )
    
    parser.add_argument('--model', required=True, help='Path to trained .pt model')
    parser.add_argument('--image', required=True, help='Path to X-ray image')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold (default: 0.3)')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold (default: 0.45)')
    parser.add_argument('--save', help='Save annotated image to path')
    parser.add_argument('--show', action='store_true', help='Display result')
    parser.add_argument('--verbose', action='store_true', help='Show all detections')
    
    args = parser.parse_args()
    
    detect_fractures(
        model_path=args.model,
        image_path=args.image,
        conf=args.conf,
        iou=args.iou,
        save_path=args.save,
        show=args.show,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()



# expect to run with:
#py -3.12 test2.py --model "best (3).pt" --image "0919_1128064861_06_WRI-R2_M004.png" --show
# Screenshot 2026-02-12 175939.png
# py -3.12 test2.py --model "best (3).pt" --image "Screenshot 2026-02-12 175939.png" --show
