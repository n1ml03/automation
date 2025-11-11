"""Item Detector Module - YOLO and Template Matching for automated detection."""

import cv2
import numpy as np
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from .agent import Agent
from .utils import get_logger

logger = get_logger(__name__)

# Conditional YOLO import
try:
    from ultralytics import YOLO  # type: ignore
    import torch
    YOLO_AVAILABLE = True
    logger.info("YOLO available")
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None  # type: ignore
    torch = None
    logger.warning("YOLO not available. Install: pip install ultralytics torch")


# ==================== YOLO DETECTOR ====================

class YOLODetector:
    """YOLO-based item detector for game objects."""

    def __init__(self, agent: Agent, model_path: str = "yolo11n.pt",
                 confidence: float = 0.25, device: str = "cpu"):
        """Initialize YOLO detector with model and configuration."""
        self.agent = agent
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None

        if not YOLO_AVAILABLE:
            logger.error(" YOLO not available")
            raise RuntimeError("YOLO not available. Install: pip install ultralytics torch")

        self._init_model()

    def _init_model(self) -> None:
        """Load YOLO model and configure device."""
        try:
            # Check YOLO is available (should be caught in __init__, but double check)
            if YOLO is None:
                raise RuntimeError("YOLO not available")
            
            logger.info(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
            # Verify model was loaded
            if self.model is None:
                raise RuntimeError("Failed to load YOLO model")

            if self.device == 'auto' and torch is not None:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'

            logger.info(f"YOLO model loaded on device: {self.device}")

        except Exception as e:
            logger.error(f"YOLO initialization failed: {e}")
            raise

    def detect(self, image: np.ndarray, conf: Optional[float] = None,
               iou: float = 0.45, imgsz: int = 640) -> List[Dict[str, Any]]:
        """Detect items in image with YOLO and extract quantities via OCR."""
        if self.model is None:
            logger.error(" YOLO model not initialized")
            return []

        if conf is None:
            conf = self.confidence

        try:
            # Run detection
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False
            )

            found_items = []

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    item_name = result.names[class_id]

                    # Extract quantity via OCR
                    quantity, ocr_text = self._extract_quantity(
                        image, (x1, y1, x2, y2)
                    )

                    found_items.append({
                        'item': item_name,
                        'quantity': quantity,
                        'x': x1,
                        'y': y1,
                        'x2': x2,
                        'y2': y2,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2,
                        'confidence': confidence,
                        'ocr_text': ocr_text
                    })

            logger.info(f"🎯 YOLO detected {len(found_items)} items")
            return found_items

        except Exception as e:
            logger.error(f" YOLO detection error: {e}")
            return []

    def _extract_quantity(self, image: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         offset_x: int = 30, offset_y: int = 0,
                         roi_width: int = 80, roi_height: int = 30) -> Tuple[int, str]:
        """Extract item quantity from region near bounding box using OCR."""
        x1, y1, x2, y2 = bbox
        img_h, img_w = image.shape[:2]

        quantity_x1 = max(0, x2 + offset_x)
        quantity_y1 = max(0, y2 + offset_y)
        quantity_x2 = min(img_w, quantity_x1 + roi_width)
        quantity_y2 = min(img_h, quantity_y1 + roi_height)

        if quantity_x1 >= quantity_x2 or quantity_y1 >= quantity_y2:
            return 0, ''

        quantity_roi = image[quantity_y1:quantity_y2, quantity_x1:quantity_x2]
        if quantity_roi.size == 0:
            return 0, ''

        try:
            if self.agent.ocr_engine is None:
                return 0, ''

            ocr_result = self.agent.ocr_engine.recognize_cv2(quantity_roi)
            ocr_text = ocr_result.get('text', '')
            quantity = self._parse_quantity(ocr_text)

            return quantity, ocr_text

        except Exception as e:
            logger.debug(f"Quantity OCR error: {e}")
            return 0, ''

    @staticmethod
    def _parse_quantity(text: str) -> int:
        """Parse quantity number from OCR text."""
        if not text:
            return 0

        text = text.strip()
        if text.startswith('x') or text.startswith('X'):
            text = text[1:].strip()
        text = text.replace(',', '').replace(' ', '')

        numbers = re.findall(r'\d+', text)
        if not numbers:
            return 0

        try:
            return int("".join(numbers))
        except ValueError:
            return 0


# ==================== TEMPLATE MATCHER ====================

class TemplateMatcher:
    """Template-based item detector using OpenCV matching."""

    def __init__(self, templates_dir: str = "templates",
                 threshold: float = 0.85, method: str = "TM_CCOEFF_NORMED"):
        """Initialize template matcher with directory and matching threshold."""
        self.templates_dir = templates_dir
        self.threshold = threshold
        self.method = method
        self.templates = self._load_templates()

        logger.info(f"TemplateMatcher initialized with {len(self.templates)} templates")

    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Load template images from directory."""
        if not os.path.isdir(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return {}

        templates = {}
        supported_formats = ['.png', '.jpg', '.jpeg']

        for filename in os.listdir(self.templates_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.templates_dir, filename)

                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        templates[name] = img
                        logger.debug(f"Loaded template: {name}")
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {e}")

        return templates

    def detect(self, image: np.ndarray,
               threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Detect items in image using template matching."""
        if threshold is None:
            threshold = self.threshold

        try:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.error(f"Grayscale conversion error: {e}")
            return []

        found_items = []

        for template_name, template in self.templates.items():
            matches = self._find_matches(image_gray, template, threshold)

            for x, y in matches:
                h, w = template.shape
                found_items.append({
                    'item': template_name,
                    'quantity': 0,
                    'x': x,
                    'y': y,
                    'x2': x + w,
                    'y2': y + h,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'confidence': 0.0,
                    'ocr_text': ''
                })

        unique_items = self._remove_duplicates(found_items, min_distance=10)
        logger.info(f"Template matching found {len(unique_items)} items")
        return unique_items

    def _find_matches(self, image_gray: np.ndarray, template: np.ndarray,
                     threshold: float) -> List[Tuple[int, int]]:
        """Find template match positions in image."""
        try:
            method = getattr(cv2, self.method)
            res = cv2.matchTemplate(image_gray, template, method)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                res = 1 - res

            locs = np.where(res >= threshold)
            matches = list(zip(*locs[::-1]))

            return matches

        except Exception as e:
            logger.error(f"Template matching error: {e}")
            return []

    @staticmethod
    def _remove_duplicates(items: List[Dict[str, Any]],
                          min_distance: int) -> List[Dict[str, Any]]:
        """Remove duplicate detections within minimum distance."""
        if not items:
            return []

        sorted_items = sorted(items, key=lambda i: (i['item'], i['x'], i['y']))
        unique_items = []

        for item in sorted_items:
            is_duplicate = False

            for existing in unique_items:
                if (existing['item'] == item['item'] and
                    abs(existing['x'] - item['x']) < min_distance and
                    abs(existing['y'] - item['y']) < min_distance):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_items.append(item)

        return unique_items


# ==================== USAGE EXAMPLES ====================

"""
# Example 1: Using YOLO Detector
from core.agent import Agent
from core.detector import YOLODetector

agent = Agent()
detector = YOLODetector(
    agent=agent,
    model_path="yolo11n.pt",
    confidence=0.25,
    device="cpu"  # or "cuda", "mps", "auto"
)

# Detect items in screenshot
screenshot = agent.snapshot()
items = detector.detect(screenshot)

for item in items:
    print(f"Found: {item['item']} x{item['quantity']} "
          f"at ({item['center_x']}, {item['center_y']}) "
          f"with confidence {item['confidence']:.2f}")


# Example 2: Using Template Matcher (fallback)
from core.detector import TemplateMatcher

matcher = TemplateMatcher(
    templates_dir="templates",
    threshold=0.85
)

screenshot = agent.snapshot()
items = matcher.detect(screenshot)

for item in items:
    print(f"Found: {item['item']} at ({item['center_x']}, {item['center_y']})")


# Example 3: Hybrid approach (thử YOLO trước, fallback về template)
try:
    detector = YOLODetector(agent)
    items = detector.detect(screenshot)
    if not items:
        # Fallback to template matching
        matcher = TemplateMatcher()
        items = matcher.detect(screenshot)
except RuntimeError:
    # YOLO không available, dùng template matching
    matcher = TemplateMatcher()
    items = matcher.detect(screenshot)
"""

