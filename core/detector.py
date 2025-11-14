"""
Item Detector Module - YOLO and Template Matching for automated detection.

This module provides three main components:
1. YOLODetector - AI-based object detection with YOLO
2. TemplateMatcher - Template-based detection with OpenCV
3. OCRTextProcessor - Advanced OCR text processing and validation

OCRTextProcessor Integration Across Automations:
================================================

FESTIVALS (automations/festivals.py):
- compare_results(): Enhanced validation with field-specific extraction
- process_festival_data(): Extract values from OCR text by field type
- validate_processed_data(): Validate processed data with type-aware logic
- Handles: 勝利点数, 推奨ランク, Sランクボーダー, 消費FP, Ｚマネー, 
  獲得EXP, drop ranges, items, rewards

GACHAS (automations/gachas.py):
- process_gacha_result(): Process rarity and character with OCR cleaning
- validate_gacha_result(): Validate gacha results with template matching
- Handles: Rarity matching (SSR, SR, R, N, ★), character name cleaning

HOPPING (automations/hopping.py):
- process_world_name(): Clean and normalize world names from OCR
- verify_hop_success(): Enhanced world name comparison with similarity detection
- Handles: World name normalization, OCR artifact handling, similarity checking

Key Features:
- Field-specific extraction (numbers, ranks, money, EXP, items)
- Drop range validation (e.g., "3 ~ 4")
- Template matching with fuzzy comparison
- OCR artifact handling (I→1, O→0, etc.)
- Confidence scoring for all extractions
- Comprehensive validation with detailed results
"""

import cv2
import numpy as np
import os
import re
from difflib import SequenceMatcher
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


# ==================== OCR TEXT PROCESSOR ====================

class OCRTextProcessor:
    """Advanced OCR text processor for game data extraction and validation."""

    @staticmethod
    def _extract_number(text: str, position: int = 0, clean_chars: Optional[List[str]] = None) -> Optional[int]:
        """
        Extract number from text at given position (0=first, -1=last).
        
        Args:
            text: Text to extract number from
            position: Position in numbers list (0=first, -1=last)
            clean_chars: List of characters to remove before extraction
            
        Returns:
            int: Extracted number or None if not found
        """
        if not text:
            return None
        
        # Clean text
        cleaned = text.strip()
        if clean_chars:
            for char in clean_chars:
                cleaned = cleaned.replace(char, '')
        else:
            cleaned = cleaned.replace(',', '').replace(' ', '')
        
        # Extract numbers
        numbers = re.findall(r'\d+', cleaned)
        if numbers:
            try:
                idx = position if position >= 0 else len(numbers) + position
                return int(numbers[idx])
            except (ValueError, IndexError):
                return None
        
        return None

    @staticmethod
    def extract_victory_points(text: str) -> Optional[int]:
        """
        Extract victory points from text.
        Example: "3ポイント先取" -> 3
        
        Args:
            text: OCR text containing victory points
            
        Returns:
            int: Extracted number or None if not found
        """
        return OCRTextProcessor._extract_number(text, position=0)

    @staticmethod
    def extract_rank(text: str) -> Optional[str]:
        """
        Extract rank letter from text.
        Example: "推奨ランク E" -> "E"
        
        Args:
            text: OCR text containing rank
            
        Returns:
            str: Extracted rank letter (E, D, C, B, A, S, SS, SSS) or None
        """
        if not text:
            return None
        
        # Clean text
        text = text.strip().upper()
        
        # Look for rank patterns (SSS, SS, S, A, B, C, D, E)
        rank_pattern = r'\b(SSS+|SSS|SS|S|A|B|C|D|E|F)\b'
        match = re.search(rank_pattern, text)
        
        if match:
            return match.group(1)
        
        return None

    @staticmethod
    def extract_s_rank_border(text: str) -> Optional[int]:
        """
        Extract S rank border score from text.
        Example: "Sランクボーダー 450" -> 450
        
        Args:
            text: OCR text containing S rank border
            
        Returns:
            int: Extracted number or None if not found
        """
        return OCRTextProcessor._extract_number(text, position=-1)

    @staticmethod
    def extract_fp_cost(text: str) -> Optional[int]:
        """
        Extract FP cost from text.
        Example: "FP 10" -> 10
        
        Args:
            text: OCR text containing FP cost
            
        Returns:
            int: Extracted number or None if not found
        """
        return OCRTextProcessor._extract_number(text, position=0)

    @staticmethod
    def extract_z_money(text: str) -> Optional[int]:
        """
        Extract Z Money (Zack Money) from text.
        Example: "獲得ザックマネー × 10300" -> 10300
        
        Args:
            text: OCR text containing Z Money
            
        Returns:
            int: Extracted number or None if not found
        """
        if not text:
            return None
        
        # Clean text - remove common OCR artifacts and join all numbers
        cleaned = text.strip().replace(',', '').replace(' ', '').replace('×', '').replace('x', '').replace('X', '')
        numbers = re.findall(r'\d+', cleaned)
        if numbers:
            try:
                # Join all numbers (in case OCR split them)
                return int(''.join(numbers))
            except ValueError:
                return None
        
        return None

    @staticmethod
    def extract_exp(text: str) -> Optional[int]:
        """
        Extract EXP from text.
        Example: "獲得EXP 158" -> 158
        
        Args:
            text: OCR text containing EXP
            
        Returns:
            int: Extracted number or None if not found
        """
        return OCRTextProcessor._extract_number(text, position=-1)

    @staticmethod
    def extract_item_quantity(text: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Extract item name and quantity from text.
        Example: "クリスマス宝箱EX(プラチナ) x2" -> ("クリスマス宝箱EX(プラチナ)", 2)
        
        Args:
            text: OCR text containing item and quantity
            
        Returns:
            Tuple[Optional[str], Optional[int]]: (item_name, quantity) or (None, None)
        """
        if not text:
            return None, None
        
        text = text.strip()
        
        # Pattern: item name followed by x/X and number
        # Example: "item_name x2" or "item_name ×2"
        pattern = r'(.+?)\s*[xX×]\s*(\d+)'
        match = re.search(pattern, text)
        
        if match:
            item_name = match.group(1).strip()
            try:
                quantity = int(match.group(2))
                return item_name, quantity
            except ValueError:
                return item_name, None
        
        # If no quantity pattern found, try to extract just numbers at the end
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                quantity = int(numbers[-1])
                # Remove the number from text to get item name
                item_name = re.sub(r'\s*[xX×]?\s*\d+\s*$', '', text).strip()
                return item_name, quantity
            except ValueError:
                pass
        
        # Return text as item name with no quantity
        return text, None

    @staticmethod
    def extract_venus_memory_quantity(text: str) -> Optional[int]:
        """
        Extract Venus Memory quantity from text (handle special characters).
        Example: "ヴィーナスメモリー x3" -> 3
        
        Args:
            text: OCR text containing venus memory quantity
            
        Returns:
            int: Extracted number or None if not found
        """
        if not text:
            return None
        
        # Clean text - remove special characters that OCR might misread
        text = text.strip()
        # Replace common OCR artifacts
        text = text.replace('o', '0')  # OCR might read 0 as o
        text = text.replace('O', '0')
        text = text.replace('l', '1')  # OCR might read 1 as l
        text = text.replace('I', '1')
        
        # Extract numbers
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                return int(numbers[-1])  # Return last number
            except ValueError:
                return None
        
        return None

    @staticmethod
    def parse_drop_range(text: str) -> Optional[Tuple[int, int]]:
        """
        Parse drop range from text.
        Example: "3 ~ 4" -> (3, 4)
        
        Args:
            text: OCR text containing drop range
            
        Returns:
            Tuple[int, int]: (min, max) range or None if not found
        """
        if not text:
            return None
        
        # Clean text
        text = text.strip()
        
        # Pattern: number ~ number or number~number or number - number
        pattern = r'(\d+)\s*[~～\-]\s*(\d+)'
        match = re.search(pattern, text)
        
        if match:
            try:
                min_val = int(match.group(1))
                max_val = int(match.group(2))
                return (min_val, max_val)
            except ValueError:
                return None
        
        # Try single number (range is same number)
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                val = int(numbers[0])
                return (val, val)
            except ValueError:
                return None
        
        return None

    @staticmethod
    def is_in_drop_range(value: int, range_text: str) -> bool:
        """
        Check if value is within drop range.
        Example: value=3, range_text="3 ~ 4" -> True
        
        Args:
            value: Value to check
            range_text: Range text (e.g., "3 ~ 4")
            
        Returns:
            bool: True if value is in range, False otherwise
        """
        drop_range = OCRTextProcessor.parse_drop_range(range_text)
        if drop_range is None:
            return False
        
        min_val, max_val = drop_range
        return min_val <= value <= max_val

    @staticmethod
    def normalize_text_for_comparison(text: str) -> str:
        """
        Normalize text for comparison (remove spaces, lowercase, remove special chars).
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove spaces
        text = text.replace(' ', '')
        text = text.replace('\u3000', '')  # Remove full-width space
        
        # Remove common punctuation (but keep essential ones)
        text = text.replace(',', '')
        text = text.replace('.', '')
        
        return text.strip()

    @staticmethod
    def compare_with_template(ocr_text: str, template_text: str, 
                            threshold: float = 0.8) -> bool:
        """
        Compare OCR text with template text using fuzzy matching with SequenceMatcher.
        
        Args:
            ocr_text: Text from OCR
            template_text: Expected template text
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            bool: True if texts match (above threshold)
        """
        if not ocr_text or not template_text:
            return False
        
        # Normalize both texts
        ocr_normalized = OCRTextProcessor.normalize_text_for_comparison(ocr_text)
        template_normalized = OCRTextProcessor.normalize_text_for_comparison(template_text)
        
        # Exact match
        if ocr_normalized == template_normalized:
            return True
        
        # Substring match
        if ocr_normalized in template_normalized or template_normalized in ocr_normalized:
            return True
        
        # Use SequenceMatcher for better similarity calculation
        if len(ocr_normalized) == 0 or len(template_normalized) == 0:
            return False
        
        similarity = SequenceMatcher(None, ocr_normalized, template_normalized).ratio()
        return similarity >= threshold

    @staticmethod
    def validate_field(field_name: str, ocr_text: str, expected_value: Any) -> Dict[str, Any]:
        """
        Validate OCR field against expected value with field-specific logic.
        
        Args:
            field_name: Name of the field (e.g., "勝利点数", "推奨ランク")
            ocr_text: Text from OCR
            expected_value: Expected value from data
            
        Returns:
            Dict with validation result:
            {
                'field': str,
                'status': str ('match', 'mismatch', 'error'),
                'extracted': Any,
                'expected': Any,
                'ocr_text': str,
                'message': str
            }
        """
        result = {
            'field': field_name,
            'status': 'error',
            'extracted': None,
            'expected': expected_value,
            'ocr_text': ocr_text,
            'message': ''
        }
        
        try:
            # Handle different field types
            if field_name == '勝利点数':
                extracted = OCRTextProcessor.extract_victory_points(ocr_text)
                expected = int(expected_value) if expected_value else None
                result['extracted'] = extracted
                result['status'] = 'match' if extracted == expected else 'mismatch'
                result['message'] = f"Victory points: {extracted} vs {expected}"
                
            elif field_name == '推奨ランク':
                extracted = OCRTextProcessor.extract_rank(ocr_text)
                expected = str(expected_value).strip().upper() if expected_value else None
                result['extracted'] = extracted
                result['status'] = 'match' if extracted == expected else 'mismatch'
                result['message'] = f"Rank: {extracted} vs {expected}"
                
            elif field_name == 'Sランクボーダー':
                extracted = OCRTextProcessor.extract_s_rank_border(ocr_text)
                expected = int(expected_value) if expected_value else None
                result['extracted'] = extracted
                result['status'] = 'match' if extracted == expected else 'mismatch'
                result['message'] = f"S rank border: {extracted} vs {expected}"
                
            elif field_name == '消費FP':
                extracted = OCRTextProcessor.extract_fp_cost(ocr_text)
                expected = int(expected_value) if expected_value else None
                result['extracted'] = extracted
                result['status'] = 'match' if extracted == expected else 'mismatch'
                result['message'] = f"FP cost: {extracted} vs {expected}"
                
            elif field_name == '獲得ザックマネー' or field_name == 'Ｚマネー':
                extracted = OCRTextProcessor.extract_z_money(ocr_text)
                expected = int(expected_value) if expected_value else None
                result['extracted'] = extracted
                result['status'] = 'match' if extracted == expected else 'mismatch'
                result['message'] = f"Z Money: {extracted} vs {expected}"
                
            elif field_name in ['獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース']:
                extracted = OCRTextProcessor.extract_exp(ocr_text)
                expected = int(expected_value) if expected_value else None
                result['extracted'] = extracted
                result['status'] = 'match' if extracted == expected else 'mismatch'
                result['message'] = f"EXP: {extracted} vs {expected}"
                
            elif '報酬' in field_name or 'クリア' in field_name:
                # Reward fields - use template matching
                match = OCRTextProcessor.compare_with_template(ocr_text, str(expected_value))
                result['extracted'] = ocr_text
                result['status'] = 'match' if match else 'mismatch'
                result['message'] = f"Template match: {match}"
                
            elif 'コイン' in field_name or 'ドロップ' in field_name:
                # Drop items - check range
                extracted = OCRTextProcessor.extract_exp(ocr_text)  # Extract number
                if extracted is not None:
                    in_range = OCRTextProcessor.is_in_drop_range(extracted, str(expected_value))
                    result['extracted'] = extracted
                    result['status'] = 'match' if in_range else 'mismatch'
                    result['message'] = f"Drop: {extracted} in range {expected_value} = {in_range}"
                else:
                    result['status'] = 'mismatch'
                    result['message'] = f"Cannot extract drop value from: {ocr_text}"
                    
            else:
                # Default: simple text comparison
                match = OCRTextProcessor.compare_with_template(ocr_text, str(expected_value))
                result['extracted'] = ocr_text
                result['status'] = 'match' if match else 'mismatch'
                result['message'] = f"Text match: {match}"
                
        except Exception as e:
            result['status'] = 'error'
            result['message'] = f"Validation error: {str(e)}"
            
        return result

    @staticmethod
    def validate_multiple_fields(extracted_data: Dict[str, str],
                                expected_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple fields at once.
        
        Args:
            extracted_data: Dictionary of field_name -> ocr_text
            expected_data: Dictionary of field_name -> expected_value
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of field_name -> validation_result
        """
        results = {}
        
        for field_name, expected_value in expected_data.items():
            if field_name not in extracted_data:
                results[field_name] = {
                    'field': field_name,
                    'status': 'missing',
                    'extracted': None,
                    'expected': expected_value,
                    'ocr_text': '',
                    'message': 'Field not found in extracted data'
                }
            else:
                ocr_text = extracted_data[field_name]
                results[field_name] = OCRTextProcessor.validate_field(
                    field_name, ocr_text, expected_value
                )
        
        return results

    @staticmethod
    def get_validation_summary(validation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from validation results.
        
        Args:
            validation_results: Results from validate_multiple_fields or validate_field
            
        Returns:
            Dict with summary:
            {
                'total': int,
                'matched': int,
                'mismatched': int,
                'missing': int,
                'errors': int,
                'match_rate': float,
                'status': str ('pass' or 'fail')
            }
        """
        total = len(validation_results)
        matched = sum(1 for r in validation_results.values() if r['status'] == 'match')
        mismatched = sum(1 for r in validation_results.values() if r['status'] == 'mismatch')
        missing = sum(1 for r in validation_results.values() if r['status'] == 'missing')
        errors = sum(1 for r in validation_results.values() if r['status'] == 'error')
        
        match_rate = matched / total if total > 0 else 0.0
        status = 'pass' if matched == total else 'fail'
        
        return {
            'total': total,
            'matched': matched,
            'mismatched': mismatched,
            'missing': missing,
            'errors': errors,
            'match_rate': match_rate,
            'status': status
        }


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
        """
        Remove duplicate detections within minimum distance.
        Optimized to check only recent items instead of all items (O(n²) -> ~O(n)).
        """
        if not items:
            return []

        # Group by item name first to reduce comparisons
        by_item = {}
        for item in items:
            name = item['item']
            if name not in by_item:
                by_item[name] = []
            by_item[name].append(item)

        unique_items = []
        # Check only last N items for duplicates (most likely duplicates are nearby)
        CHECK_WINDOW = 10

        for name, item_list in by_item.items():
            # Sort by position for spatial locality
            sorted_items = sorted(item_list, key=lambda i: (i['x'], i['y']))

            for item in sorted_items:
                # Check only against recent additions (spatial locality optimization)
                is_duplicate = any(
                    existing['item'] == item['item'] and
                    abs(existing['x'] - item['x']) < min_distance and
                    abs(existing['y'] - item['y']) < min_distance
                    for existing in unique_items[-CHECK_WINDOW:]
                )
                if not is_duplicate:
                    unique_items.append(item)

        return unique_items

