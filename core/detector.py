"""
Item Detector Module - YOLO and Template Matching for automated detection.

This module provides three main components:
1. YOLODetector - AI-based object detection with YOLO
2. TemplateMatcher - Template-based detection with OpenCV
3. OCRTextProcessor - Advanced OCR text processing and validation

"""

import cv2
import numpy as np
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
from functools import lru_cache
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


# ==================== DATA CLASSES ====================

@dataclass
class DetectionResult:
    """Result from YOLO or Template detection."""
    item: str
    quantity: int = 0
    x: int = 0
    y: int = 0
    x2: int = 0
    y2: int = 0
    center_x: int = 0
    center_y: int = 0
    confidence: float = 0.0
    ocr_text: str = ''
    
    def __post_init__(self):
        """Calculate center if not provided."""
        if self.center_x == 0 and self.center_y == 0:
            self.center_x = (self.x + self.x2) // 2
            self.center_y = (self.y + self.y2) // 2


@dataclass
class ExtractionResult:
    """Result from field extraction."""
    value: Any
    raw_text: str
    confidence: float
    success: bool
    error_message: str = ''


@dataclass
class ValidationResult:
    """Result from field validation."""
    field: str
    status: str  # 'match', 'mismatch', 'error', 'missing'
    extracted: Any
    expected: Any
    ocr_text: str
    message: str
    confidence: float = 0.0


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total: int
    matched: int
    mismatched: int
    missing: int
    errors: int
    match_rate: float
    status: str  # 'pass' or 'fail'


# ==================== TEXT PROCESSING UTILITIES ====================

class TextProcessor:
    """Unified text processing utilities - NO DUPLICATION."""
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def normalize_text(text: str, remove_spaces: bool = True, 
                      lowercase: bool = True) -> str:
        """
        Normalize text for comparison with caching.
        
        Args:
            text: Text to normalize
            remove_spaces: Remove all spaces
            lowercase: Convert to lowercase
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        result = text.strip()
        
        if lowercase:
            result = result.lower()
        
        if remove_spaces:
            result = result.replace(' ', '').replace('\u3000', '')
        
        # Remove common punctuation
        result = result.replace(',', '').replace('.', '')
        
        return result
    
    @staticmethod
    def clean_ocr_artifacts(text: str) -> str:
        """
        Clean common OCR artifacts.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Common OCR mistakes
        replacements = {
            'o': '0', 'O': '0',  # Letter O to zero
            'l': '1', 'I': '1',  # Letter I/l to one
        }
        
        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result.strip()
    
    @staticmethod
    def extract_numbers(text: str, clean_chars: Optional[List[str]] = None) -> List[int]:
        """
        Extract all numbers from text.
        
        Args:
            text: Text to extract from
            clean_chars: Characters to remove before extraction
            
        Returns:
            List[int]: List of extracted numbers
        """
        if not text:
            return []
        
        # Clean text
        cleaned = text.strip()
        if clean_chars:
            for char in clean_chars:
                cleaned = cleaned.replace(char, '')
        else:
            cleaned = cleaned.replace(',', '').replace(' ', '')
        
        # Extract numbers
        numbers = re.findall(r'\d+', cleaned)
        try:
            return [int(n) for n in numbers]
        except ValueError:
            return []
    
    @staticmethod
    def get_number_at_position(text: str, position: int = 0, 
                              clean_chars: Optional[List[str]] = None) -> Optional[int]:
        """
        Get number at specific position in text.
        
        Args:
            text: Text to extract from
            position: Position in numbers list (0=first, -1=last)
            clean_chars: Characters to remove before extraction
            
        Returns:
            Optional[int]: Number at position or None
        """
        numbers = TextProcessor.extract_numbers(text, clean_chars)
        if not numbers:
            return None
        
        try:
            idx = position if position >= 0 else len(numbers) + position
            return numbers[idx]
        except IndexError:
            return None
    
    @staticmethod
    @lru_cache(maxsize=512)
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (cached).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity ratio (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    @staticmethod
    def fuzzy_match(text: str, template: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy match text against template.
        
        Args:
            text: Text to match
            template: Template to match against
            threshold: Similarity threshold
            
        Returns:
            bool: True if match
        """
        if not text or not template:
            return False
        
        # Normalize both
        text_norm = TextProcessor.normalize_text(text)
        template_norm = TextProcessor.normalize_text(template)
        
        # Exact match
        if text_norm == template_norm:
            return True
        
        # Substring match
        if text_norm in template_norm or template_norm in text_norm:
            return True
        
        # Similarity match
        if len(text_norm) == 0 or len(template_norm) == 0:
            return False
        
        similarity = TextProcessor.calculate_similarity(text_norm, template_norm)
        return similarity >= threshold


# ==================== FIELD EXTRACTORS (STRATEGY PATTERN) ====================

class FieldExtractor(Protocol):
    """Protocol for field extractors."""
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract field value from text."""
        ...


class NumberExtractor:
    """Extract number from text."""
    
    def __init__(self, position: int = 0, clean_chars: Optional[List[str]] = None):
        self.position = position
        self.clean_chars = clean_chars
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract number at position."""
        value = TextProcessor.get_number_at_position(text, self.position, self.clean_chars)
        
        return ExtractionResult(
            value=value,
            raw_text=text,
            confidence=0.9 if value is not None else 0.0,
            success=value is not None
        )


class RankExtractor:
    """Extract rank letter from text."""
    
    RANK_PATTERN = r'\b(SSS+|SSS|SS|S|A|B|C|D|E|F)\b'
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract rank."""
        if not text:
            return ExtractionResult(None, text, 0.0, False)
        
        text_upper = text.strip().upper()
        match = re.search(self.RANK_PATTERN, text_upper)
        
        if match:
            rank = match.group(1)
            return ExtractionResult(rank, text, 1.0, True)
        
        return ExtractionResult(None, text, 0.0, False)


class MoneyExtractor:
    """Extract money/currency from text."""
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract money (join all numbers)."""
        if not text:
            return ExtractionResult(None, text, 0.0, False)
        
        # Clean and join all numbers
        cleaned = text.strip().replace(',', '').replace(' ', '')
        cleaned = cleaned.replace('×', '').replace('x', '').replace('X', '')
        
        numbers = re.findall(r'\d+', cleaned)
        if numbers:
            try:
                value = int(''.join(numbers))
                return ExtractionResult(value, text, 0.9, True)
            except ValueError:
                pass
        
        return ExtractionResult(None, text, 0.0, False)


class ItemQuantityExtractor:
    """Extract item name and quantity."""
    
    PATTERN = r'(.+?)\s*[xX×]\s*(\d+)'
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract item and quantity."""
        if not text:
            return ExtractionResult((None, None), text, 0.0, False)
        
        text = text.strip()
        
        # Try pattern match
        match = re.search(self.PATTERN, text)
        if match:
            item_name = match.group(1).strip()
            try:
                quantity = int(match.group(2))
                return ExtractionResult((item_name, quantity), text, 0.9, True)
            except ValueError:
                pass
        
        # Fallback: try to extract numbers at end
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                quantity = int(numbers[-1])
                item_name = re.sub(r'\s*[xX×]?\s*\d+\s*$', '', text).strip()
                return ExtractionResult((item_name, quantity), text, 0.7, True)
            except ValueError:
                pass
        
        # No quantity found
        return ExtractionResult((text, None), text, 0.5, True)


class DropRangeExtractor:
    """Extract drop range (e.g., '3 ~ 4')."""
    
    PATTERN = r'(\d+)\s*[~～\-]\s*(\d+)'
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract drop range."""
        if not text:
            return ExtractionResult(None, text, 0.0, False)
        
        text = text.strip()
        
        # Try range pattern
        match = re.search(self.PATTERN, text)
        if match:
            try:
                min_val = int(match.group(1))
                max_val = int(match.group(2))
                return ExtractionResult((min_val, max_val), text, 0.9, True)
            except ValueError:
                pass
        
        # Single number (range = same number)
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                val = int(numbers[0])
                return ExtractionResult((val, val), text, 0.8, True)
            except ValueError:
                pass
        
        return ExtractionResult(None, text, 0.0, False)


# ==================== OCR TEXT PROCESSOR (REFACTORED) ====================

class OCRTextProcessor:
    """
    Advanced OCR text processor with strategy pattern.
    Now uses extractors instead of dozens of static methods.
    """
    
    # Field extractors registry
    EXTRACTORS: Dict[str, FieldExtractor] = {
        '勝利点数': NumberExtractor(position=0),
        '推奨ランク': RankExtractor(),
        'Sランクボーダー': NumberExtractor(position=-1),
        '消費FP': NumberExtractor(position=0),
        '獲得ザックマネー': MoneyExtractor(),
        'Ｚマネー': MoneyExtractor(),
        '獲得EXP-Ace': NumberExtractor(position=-1),
        '獲得EXP-NonAce': NumberExtractor(position=-1),
        'エース': NumberExtractor(position=-1),
        '非エース': NumberExtractor(position=-1),
        'item_quantity': ItemQuantityExtractor(),
        'drop_range': DropRangeExtractor(),
    }
    
    @classmethod
    def extract_field(cls, field_name: str, text: str) -> ExtractionResult:
        """
        Extract field value using appropriate extractor.
        
        Args:
            field_name: Field name
            text: OCR text
            
        Returns:
            ExtractionResult: Extraction result
        """
        # Get extractor
        extractor = cls.EXTRACTORS.get(field_name)
        
        if extractor is None:
            # Default: return text as-is
            return ExtractionResult(
                value=text,
                raw_text=text,
                confidence=0.5,
                success=bool(text)
            )
        
        return extractor.extract(text)
    
    @staticmethod
    def validate_field(field_name: str, ocr_text: str, 
                      expected_value: Any) -> ValidationResult:
        """
        Validate OCR field against expected value.
        
        Args:
            field_name: Field name
            ocr_text: OCR text
            expected_value: Expected value
            
        Returns:
            ValidationResult: Validation result
        """
        try:
            # Extract value using extractor
            extraction = OCRTextProcessor.extract_field(field_name, ocr_text)
            
            if not extraction.success:
                return ValidationResult(
                    field=field_name,
                    status='error',
                    extracted=None,
                    expected=expected_value,
                    ocr_text=ocr_text,
                    message=f"Failed to extract: {extraction.error_message}",
                    confidence=0.0
                )
            
            extracted_value = extraction.value
            
            # Validate based on field type
            if '報酬' in field_name or 'クリア' in field_name:
                # Reward fields - use fuzzy matching
                match = TextProcessor.fuzzy_match(ocr_text, str(expected_value))
                return ValidationResult(
                    field=field_name,
                    status='match' if match else 'mismatch',
                    extracted=ocr_text,
                    expected=expected_value,
                    ocr_text=ocr_text,
                    message=f"Template match: {match}",
                    confidence=extraction.confidence
                )
            
            elif 'コイン' in field_name or 'ドロップ' in field_name:
                # Drop items - check range
                drop_range = DropRangeExtractor().extract(str(expected_value))
                if drop_range.success and isinstance(drop_range.value, tuple):
                    min_val, max_val = drop_range.value
                    if isinstance(extracted_value, int):
                        in_range = min_val <= extracted_value <= max_val
                        return ValidationResult(
                            field=field_name,
                            status='match' if in_range else 'mismatch',
                            extracted=extracted_value,
                            expected=expected_value,
                            ocr_text=ocr_text,
                            message=f"Drop: {extracted_value} in range [{min_val}, {max_val}] = {in_range}",
                            confidence=extraction.confidence
                        )
            
            else:
                # Direct comparison
                match = extracted_value == expected_value
                return ValidationResult(
                    field=field_name,
                    status='match' if match else 'mismatch',
                    extracted=extracted_value,
                    expected=expected_value,
                    ocr_text=ocr_text,
                    message=f"Comparison: {extracted_value} == {expected_value} = {match}",
                    confidence=extraction.confidence
                )
            
            # Default fallback (should not reach here)
            return ValidationResult(
                field=field_name,
                status='mismatch',
                extracted=extracted_value,
                expected=expected_value,
                ocr_text=ocr_text,
                message="Validation completed without specific match",
                confidence=extraction.confidence
            )
        
        except Exception as e:
            return ValidationResult(
                field=field_name,
                status='error',
                extracted=None,
                expected=expected_value,
                ocr_text=ocr_text,
                message=f"Validation error: {str(e)}",
                confidence=0.0
            )
    
    @staticmethod
    def validate_multiple_fields(extracted_data: Dict[str, str],
                                expected_data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Validate multiple fields at once.
        
        Args:
            extracted_data: Dict of field_name -> ocr_text
            expected_data: Dict of field_name -> expected_value
            
        Returns:
            Dict[str, ValidationResult]: Validation results
        """
        results = {}
        
        for field_name, expected_value in expected_data.items():
            if field_name not in extracted_data:
                results[field_name] = ValidationResult(
                    field=field_name,
                    status='missing',
                    extracted=None,
                    expected=expected_value,
                    ocr_text='',
                    message='Field not found in extracted data',
                    confidence=0.0
                )
            else:
                ocr_text = extracted_data[field_name]
                results[field_name] = OCRTextProcessor.validate_field(
                    field_name, ocr_text, expected_value
                )
        
        return results
    
    @staticmethod
    def get_validation_summary(validation_results: Dict[str, ValidationResult]) -> ValidationSummary:
        """
        Get summary statistics from validation results.
        
        Args:
            validation_results: Validation results
            
        Returns:
            ValidationSummary: Summary statistics
        """
        total = len(validation_results)
        matched = sum(1 for r in validation_results.values() if r.status == 'match')
        mismatched = sum(1 for r in validation_results.values() if r.status == 'mismatch')
        missing = sum(1 for r in validation_results.values() if r.status == 'missing')
        errors = sum(1 for r in validation_results.values() if r.status == 'error')
        
        match_rate = matched / total if total > 0 else 0.0
        status = 'pass' if matched == total else 'fail'
        
        return ValidationSummary(
            total=total,
            matched=matched,
            mismatched=mismatched,
            missing=missing,
            errors=errors,
            match_rate=match_rate,
            status=status
        )
    
    # Backward compatibility methods (delegate to extractors)
    @staticmethod
    def extract_victory_points(text: str) -> Optional[int]:
        """Extract victory points (backward compatibility)."""
        result = OCRTextProcessor.extract_field('勝利点数', text)
        return result.value if result.success else None
    
    @staticmethod
    def extract_rank(text: str) -> Optional[str]:
        """Extract rank (backward compatibility)."""
        result = OCRTextProcessor.extract_field('推奨ランク', text)
        return result.value if result.success else None
    
    @staticmethod
    def extract_s_rank_border(text: str) -> Optional[int]:
        """Extract S rank border (backward compatibility)."""
        result = OCRTextProcessor.extract_field('Sランクボーダー', text)
        return result.value if result.success else None
    
    @staticmethod
    def extract_fp_cost(text: str) -> Optional[int]:
        """Extract FP cost (backward compatibility)."""
        result = OCRTextProcessor.extract_field('消費FP', text)
        return result.value if result.success else None
    
    @staticmethod
    def extract_z_money(text: str) -> Optional[int]:
        """Extract Z Money (backward compatibility)."""
        result = OCRTextProcessor.extract_field('獲得ザックマネー', text)
        return result.value if result.success else None
    
    @staticmethod
    def extract_exp(text: str) -> Optional[int]:
        """Extract EXP (backward compatibility)."""
        result = OCRTextProcessor.extract_field('獲得EXP-Ace', text)
        return result.value if result.success else None
    
    @staticmethod
    def extract_item_quantity(text: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract item and quantity (backward compatibility)."""
        result = OCRTextProcessor.extract_field('item_quantity', text)
        if result.success and isinstance(result.value, tuple):
            return result.value
        return (None, None)
    
    @staticmethod
    def extract_venus_memory_quantity(text: str) -> Optional[int]:
        """Extract Venus Memory quantity (backward compatibility)."""
        if not text:
            return None
        # Clean OCR artifacts then extract number
        cleaned = TextProcessor.clean_ocr_artifacts(text)
        numbers = TextProcessor.extract_numbers(cleaned)
        return numbers[-1] if numbers else None
    
    @staticmethod
    def parse_drop_range(text: str) -> Optional[Tuple[int, int]]:
        """Parse drop range (backward compatibility)."""
        result = OCRTextProcessor.extract_field('drop_range', text)
        return result.value if result.success else None
    
    @staticmethod
    def is_in_drop_range(value: int, range_text: str) -> bool:
        """Check if value is in drop range."""
        drop_range = OCRTextProcessor.parse_drop_range(range_text)
        if drop_range is None:
            return False
        min_val, max_val = drop_range
        return min_val <= value <= max_val
    
    @staticmethod
    def normalize_text_for_comparison(text: str) -> str:
        """Normalize text for comparison (delegate to TextProcessor)."""
        return TextProcessor.normalize_text(text)
    
    @staticmethod
    def compare_with_template(ocr_text: str, template_text: str, 
                            threshold: float = 0.8) -> bool:
        """Compare with template (delegate to TextProcessor)."""
        return TextProcessor.fuzzy_match(ocr_text, template_text, threshold)


# ==================== YOLO DETECTOR ====================

class YOLODetector:
    """YOLO-based item detector (optimized with TextProcessor)."""

    def __init__(self, agent: Agent, model_path: str = "yolo11n.pt",
                 confidence: float = 0.25, device: str = "cpu"):
        """Initialize YOLO detector."""
        self.agent = agent
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None

        if not YOLO_AVAILABLE:
            logger.error("YOLO not available")
            raise RuntimeError("YOLO not available. Install: pip install ultralytics torch")

        self._init_model()

    def _init_model(self) -> None:
        """Load YOLO model and configure device."""
        try:
            if YOLO is None:
                raise RuntimeError("YOLO not available")
            
            logger.info(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
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
               iou: float = 0.45, imgsz: int = 640) -> List[DetectionResult]:
        """
        Detect items in image with YOLO.
        Now returns DetectionResult dataclass.
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
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

                    # Use dataclass
                    found_items.append(DetectionResult(
                        item=item_name,
                        quantity=quantity,
                        x=x1,
                        y=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        ocr_text=ocr_text
                    ))

            logger.info(f"🎯 YOLO detected {len(found_items)} items")
            return found_items

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

    def _extract_quantity(self, image: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         offset_x: int = 30, offset_y: int = 0,
                         roi_width: int = 80, roi_height: int = 30) -> Tuple[int, str]:
        """Extract item quantity using OCR (reuses TextProcessor)."""
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

            ocr_result = self.agent.ocr_engine.recognize(quantity_roi)
            ocr_text = ocr_result.get('text', '')
            
            # Use TextProcessor for parsing (NO DUPLICATION)
            quantity = self._parse_quantity_text(ocr_text)

            return quantity, ocr_text

        except Exception as e:
            logger.debug(f"Quantity OCR error: {e}")
            return 0, ''

    @staticmethod
    def _parse_quantity_text(text: str) -> int:
        """Parse quantity from text (uses TextProcessor - NO DUPLICATION)."""
        if not text:
            return 0

        text = text.strip()
        # Remove x/X prefix
        if text.startswith('x') or text.startswith('X'):
            text = text[1:].strip()
        
        # Use TextProcessor to extract number
        numbers = TextProcessor.extract_numbers(text)
        if numbers:
            return numbers[0]
        
        return 0


# ==================== TEMPLATE MATCHER ====================

class TemplateMatcher:
    """Template-based item detector (optimized duplicate removal)."""

    def __init__(self, templates_dir: str = "templates",
                 threshold: float = 0.85, method: str = "TM_CCOEFF_NORMED"):
        """Initialize template matcher."""
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
               threshold: Optional[float] = None) -> List[DetectionResult]:
        """
        Detect items using template matching.
        Now returns DetectionResult dataclass.
        """
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
                # Use dataclass
                found_items.append(DetectionResult(
                    item=template_name,
                    quantity=0,
                    x=x,
                    y=y,
                    x2=x + w,
                    y2=y + h
                ))

        # Optimized duplicate removal
        unique_items = self._remove_duplicates_optimized(found_items, min_distance=10)
        logger.info(f"Template matching found {len(unique_items)} items")
        return unique_items

    def _find_matches(self, image_gray: np.ndarray, template: np.ndarray,
                     threshold: float) -> List[Tuple[int, int]]:
        """Find template match positions."""
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
    def _remove_duplicates_optimized(items: List[DetectionResult],
                                    min_distance: int) -> List[DetectionResult]:
        """
        Remove duplicate detections (optimized algorithm).
        Groups by item name first, then uses spatial locality.
        """
        if not items:
            return []

        # Group by item name first
        by_item: Dict[str, List[DetectionResult]] = {}
        for item in items:
            if item.item not in by_item:
                by_item[item.item] = []
            by_item[item.item].append(item)

        unique_items = []
        CHECK_WINDOW = 10  # Only check recent items

        for name, item_list in by_item.items():
            # Sort by position for spatial locality
            sorted_items = sorted(item_list, key=lambda i: (i.x, i.y))

            for item in sorted_items:
                # Check only against recent additions (spatial locality)
                is_duplicate = any(
                    existing.item == item.item and
                    abs(existing.x - item.x) < min_distance and
                    abs(existing.y - item.y) < min_distance
                    for existing in unique_items[-CHECK_WINDOW:]
                )
                if not is_duplicate:
                    unique_items.append(item)

        return unique_items
