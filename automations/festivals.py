"""
Festival Automation

Flow chuẩn:
1. touch(Template("tpl_festival.png"))
2. touch(Template("tpl_event.png"))
3. snapshot -> lưu folder (rank_E_stage_1/01_before_touch.png)
4. find_and_touch (OCR -> tìm text -> touch)
5. snapshot -> lưu folder (rank_E_stage_1/02_after_touch.png)
6. ROI scan -> so sánh CSV -> ghi OK/NG
7. touch(Template("tpl_challenge.png"))
8. touch(Template("tpl_ok.png"))
9. touch(Template("tpl_allskip.png"))
10. touch(Template("tpl_ok.png"))
11. touch(Template("tpl_result.png"))
12. snapshot -> lưu folder (rank_E_stage_1/03_result.png)
13. ROI scan -> so sánh CSV -> ghi OK/NG
14. touch(Template("tpl_ok.png")) if exist
15. touch(Template("tpl_ok.png")) if exist
16. Lặp lại
"""

import os
import time
import cv2
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from airtest.core.api import Template
from core.agent import Agent
from core.utils import get_logger, ensure_directory
from core.data import ResultWriter, load_json, load_csv
from core.config import (
    FESTIVALS_ROI_CONFIG, get_festivals_roi_config,
    FESTIVAL_CONFIG, get_festival_config, merge_config
)
from core.detector import YOLODetector, TemplateMatcher, YOLO_AVAILABLE

logger = get_logger(__name__)


class FestivalAutomation:
    """Tự động hóa Festival - chỉ giữ các bước cần thiết."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        self.agent = agent
        
        # Merge config: base config từ FESTIVAL_CONFIG + custom config
        base_config = get_festival_config()
        cfg = merge_config(base_config, config) if config else base_config

        # Paths
        self.templates_path = cfg.get('templates_path')
        self.snapshot_dir = cfg.get('snapshot_dir')
        self.results_dir = cfg.get('results_dir')
        
        # Timing
        self.wait_after_touch = cfg.get('wait_after_touch')

        # Ensure directories exist
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)

        # Initialize detector (YOLO hoặc Template Matching)
        self.detector = None
        self.use_detector = cfg.get('use_detector')
        
        if self.use_detector:
            detector_type = cfg.get('detector_type', 'template')  # 'yolo', 'template', 'auto'
            
            if detector_type == 'auto':
                # Tự động chọn: ưu tiên YOLO, fallback Template
                if YOLO_AVAILABLE:
                    try:
                        yolo_config = cfg.get('yolo_config', {})
                        self.detector = YOLODetector(
                            agent=agent,
                            model_path=yolo_config.get('model_path', 'yolo11n.pt'),
                            confidence=yolo_config.get('confidence', 0.25),
                            device=yolo_config.get('device', 'cpu')
                        )
                        logger.info("✅ Using YOLO Detector")
                    except Exception as e:
                        logger.warning(f"⚠️ YOLO init failed: {e}, fallback to Template")
                        template_config = cfg.get('template_config', {})
                        self.detector = TemplateMatcher(
                            templates_dir=template_config.get('templates_dir', self.templates_path),
                            threshold=template_config.get('threshold', 0.85)
                        )
                else:
                    template_config = cfg.get('template_config', {})
                    self.detector = TemplateMatcher(
                        templates_dir=template_config.get('templates_dir', self.templates_path),
                        threshold=template_config.get('threshold', 0.85)
                    )
                    logger.info("✅ Using Template Matcher")
            
            elif detector_type == 'yolo':
                yolo_config = cfg.get('yolo_config', {})
                self.detector = YOLODetector(
                    agent=agent,
                    model_path=yolo_config.get('model_path', 'yolo11n.pt'),
                    confidence=yolo_config.get('confidence', 0.25),
                    device=yolo_config.get('device', 'cpu')
                )
                logger.info("✅ Using YOLO Detector")
            
            elif detector_type == 'template':
                template_config = cfg.get('template_config', {})
                self.detector = TemplateMatcher(
                    templates_dir=template_config.get('templates_dir', self.templates_path),
                    threshold=template_config.get('threshold', 0.85)
                )
                logger.info("✅ Using Template Matcher")

        logger.info("FestivalAutomation initialized")

    def touch_template(self, template_name: str, optional: bool = False) -> bool:
        """Touch vào template image."""
        try:
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path):
                return optional

            if self.agent.device is None:
                logger.error("Device not connected")
                return False

            template = Template(template_path)
            pos = self.agent.device.exists(template)

            if pos:
                self.agent.device.touch(pos)
                logger.info(f"✓ {template_name}")
                time.sleep(self.wait_after_touch)
                return True

            return optional if optional else False

        except Exception as e:
            logger.error(f"✗ {template_name}: {e}") if not optional else None
            return optional

    def snapshot_and_save(self, folder_name: str, filename: str) -> Optional[Any]:
        """Chụp màn hình và lưu vào folder."""
        try:
            screenshot = self.agent.snapshot()
            if screenshot is None:
                return None

            folder_path = os.path.join(self.snapshot_dir, folder_name)
            ensure_directory(folder_path)
            file_path = os.path.join(folder_path, filename)
            cv2.imwrite(file_path, screenshot)
            logger.info(f"✓ Saved: {filename}")
            return screenshot

        except Exception as e:
            logger.error(f"✗ Snapshot: {e}")
            return None

    def snapshot_and_ocr(self) -> List[Dict[str, Any]]:
        """Chụp màn hình và OCR để lấy text + tọa độ."""
        try:
            screenshot = self.agent.snapshot()
            if screenshot is None:
                return []

            if self.agent.ocr_engine is None:
                logger.error("OCR engine not initialized")
                return []

            ocr_results = self.agent.ocr_engine.recognize_cv2(screenshot)
            lines = ocr_results.get('lines', [])

            results = []
            for line in lines:
                text = line.get('text', '').strip()
                bbox = line.get('bounding_rect', {})
                if text and bbox:
                    center_x = (bbox.get('x1', 0) + bbox.get('x3', 0)) / 2
                    center_y = (bbox.get('y1', 0) + bbox.get('y3', 0)) / 2
                    results.append({'text': text, 'center': (center_x, center_y)})

            logger.info(f"OCR: {len(results)} texts")
            return results

        except Exception as e:
            logger.error(f"✗ OCR: {e}")
            return []

    def find_text(self, ocr_results: List[Dict[str, Any]], search_text: str) -> Optional[Dict[str, Any]]:
        """Tìm text trong kết quả OCR."""
        search_lower = search_text.lower().strip()
        for result in ocr_results:
            if search_lower in result['text'].lower():
                return result
        return None

    def find_and_touch(self, search_text: str) -> bool:
        """Tìm text bằng OCR và touch."""
        logger.info(f"Find & touch: {search_text}")
        ocr_results = self.snapshot_and_ocr()
        text_info = self.find_text(ocr_results, search_text)

        if text_info:
            logger.info(f"✓ Found: {search_text}")
            success = self.agent.safe_touch(text_info['center'])
            if success:
                time.sleep(self.wait_after_touch)
            return success

        logger.warning(f"✗ Not found: {search_text}")
        return False

    def ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> str:
        """
        OCR một vùng ROI cụ thể.

        Args:
            roi_name: Tên ROI trong FESTIVALS_ROI_CONFIG
            screenshot: Screenshot để OCR, None = chụp mới

        Returns:
            str: Text được OCR từ vùng ROI (đã clean)
        """
        try:
            # Lấy config ROI
            roi_config = get_festivals_roi_config(roi_name)
            coords = roi_config['coords']  # [x1, x2, y1, y2]

            # Convert sang format (x1, y1, x2, y2) cho snapshot_region
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # Chụp hoặc crop vùng ROI
            if screenshot is None:
                roi_image = self.agent.snapshot_region(region)
            else:
                # Crop từ screenshot có sẵn
                roi_image = screenshot[y1:y2, x1:x2]

            if roi_image is None:
                logger.warning(f"✗ ROI '{roi_name}': Cannot get image")
                return ""

            # OCR vùng ROI
            if self.agent.ocr_engine is None:
                logger.error("OCR engine not initialized")
                return ""

            ocr_result = self.agent.ocr_engine.recognize_cv2(roi_image)
            text = ocr_result.get('text', '').strip()

            # Clean text
            text = self._clean_ocr_text(text)

            logger.debug(f"ROI '{roi_name}': '{text}'")
            return text

        except Exception as e:
            logger.error(f"✗ OCR ROI '{roi_name}': {e}")
            return ""

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text (loại bỏ ký tự lạ, normalize).

        Args:
            text: Text cần clean

        Returns:
            str: Text đã clean
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove newlines
        text = text.replace('\n', ' ').replace('\r', '')

        return text.strip()

    def detect_and_ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> Dict[str, Any]:
        """
        Detect object trong ROI và OCR để lấy text/quantity.
        Kết hợp detector (YOLO/Template) với OCR để verify chính xác hơn.

        Args:
            roi_name: Tên ROI trong FESTIVALS_ROI_CONFIG
            screenshot: Screenshot để scan, None = chụp mới

        Returns:
            Dict[str, Any]: 
            {
                'roi_name': str,
                'text': str,  # Text từ OCR
                'detected': bool,  # Có detect được object không
                'detections': List[Dict],  # Danh sách objects detected
                'detection_count': int,  # Số lượng objects detected
                'has_quantity': bool,  # Có quantity không (từ detector)
                'quantity': int,  # Quantity nếu có (từ detector)
            }
        """
        result = {
            'roi_name': roi_name,
            'text': '',
            'detected': False,
            'detections': [],
            'detection_count': 0,
            'has_quantity': False,
            'quantity': 0
        }

        try:
            # Lấy config ROI
            roi_config = get_festivals_roi_config(roi_name)
            coords = roi_config['coords']  # [x1, x2, y1, y2]
            x1, x2, y1, y2 = coords

            # Lấy screenshot nếu chưa có
            if screenshot is None:
                screenshot = self.agent.snapshot()
                if screenshot is None:
                    logger.warning(f"✗ Cannot get screenshot for ROI '{roi_name}'")
                    return result

            # Crop vùng ROI
            roi_image = screenshot[y1:y2, x1:x2]
            if roi_image is None or roi_image.size == 0:
                logger.warning(f"✗ ROI '{roi_name}': Invalid crop")
                return result

            # Bước 1: OCR truyền thống (luôn chạy)
            if self.agent.ocr_engine is not None:
                ocr_result = self.agent.ocr_engine.recognize_cv2(roi_image)
                text = self._clean_ocr_text(ocr_result.get('text', ''))
                result['text'] = text
                logger.debug(f"ROI '{roi_name}' OCR: '{text}'")

            # Bước 2: Detection (nếu có detector)
            if self.detector is not None:
                detections = self.detector.detect(roi_image)
                result['detections'] = detections
                result['detection_count'] = len(detections)
                result['detected'] = len(detections) > 0

                # Log detection results
                if detections:
                    logger.debug(f"ROI '{roi_name}' detected {len(detections)} objects:")
                    for det in detections:
                        item_name = det.get('item', 'unknown')
                        quantity = det.get('quantity', 0)
                        confidence = det.get('confidence', 0)
                        logger.debug(f"  - {item_name} x{quantity} (conf: {confidence:.2f})")
                        
                        # Lưu quantity từ detection đầu tiên
                        if quantity > 0 and not result['has_quantity']:
                            result['has_quantity'] = True
                            result['quantity'] = quantity

            logger.info(f"✓ ROI '{roi_name}': text='{result['text']}', "
                       f"detected={result['detected']} ({result['detection_count']} objects), "
                       f"quantity={result['quantity'] if result['has_quantity'] else 'N/A'}")

            return result

        except Exception as e:
            logger.error(f"✗ Detect & OCR ROI '{roi_name}': {e}")
            return result

    def scan_screen_roi(self, screenshot: Optional[Any] = None,
                       roi_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scan màn hình theo các vùng ROI đã định nghĩa.

        Args:
            screenshot: Screenshot để scan, None = chụp mới
            roi_names: Danh sách tên ROI cần scan, None = scan tất cả

        Returns:
            Dict[str, Any]: Dictionary với key là tên ROI, value là text OCR được
        """
        try:
            # Lấy screenshot nếu chưa có
            if screenshot is None:
                screenshot = self.agent.snapshot()
                if screenshot is None:
                    logger.error("Cannot get screenshot")
                    return {}

            # Xác định danh sách ROI cần scan
            if roi_names is None:
                roi_names = list(FESTIVALS_ROI_CONFIG.keys())

            # OCR từng ROI
            results = {}
            for roi_name in roi_names:
                try:
                    text = self.ocr_roi(roi_name, screenshot)
                    results[roi_name] = text
                    logger.debug(f"✓ {roi_name}: '{text}'")
                except Exception as e:
                    logger.warning(f"✗ {roi_name}: {e}")
                    results[roi_name] = ""

            logger.info(f"Scanned {len(results)} ROIs")
            return results

        except Exception as e:
            logger.error(f"✗ Scan screen ROI: {e}")
            return {}

    def scan_screen_roi_with_detector(self, screenshot: Optional[Any] = None,
                                     roi_names: Optional[List[str]] = None,
                                     use_detector: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Scan màn hình với detector (YOLO/Template) + OCR.
        Version nâng cao của scan_screen_roi với detection info.

        Args:
            screenshot: Screenshot để scan, None = chụp mới
            roi_names: Danh sách tên ROI cần scan, None = scan tất cả
            use_detector: Có sử dụng detector không (True = detect + OCR, False = chỉ OCR)

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary với key là tên ROI, value là detection result
            {
                'roi_name': {
                    'text': str,
                    'detected': bool,
                    'detections': List[Dict],
                    'detection_count': int,
                    'has_quantity': bool,
                    'quantity': int
                }
            }
        """
        try:
            # Lấy screenshot nếu chưa có
            if screenshot is None:
                screenshot = self.agent.snapshot()
                if screenshot is None:
                    logger.error("Cannot get screenshot")
                    return {}

            # Xác định danh sách ROI cần scan
            if roi_names is None:
                roi_names = list(FESTIVALS_ROI_CONFIG.keys())

            # Scan từng ROI với detector
            results = {}
            for roi_name in roi_names:
                try:
                    if use_detector and self.detector is not None:
                        # Sử dụng detector + OCR
                        result = self.detect_and_ocr_roi(roi_name, screenshot)
                        results[roi_name] = result
                    else:
                        # Fallback: chỉ OCR
                        text = self.ocr_roi(roi_name, screenshot)
                        results[roi_name] = {
                            'roi_name': roi_name,
                            'text': text,
                            'detected': False,
                            'detections': [],
                            'detection_count': 0,
                            'has_quantity': False,
                            'quantity': 0
                        }
                except Exception as e:
                    logger.warning(f"✗ {roi_name}: {e}")
                    results[roi_name] = {
                        'roi_name': roi_name,
                        'text': '',
                        'detected': False,
                        'detections': [],
                        'detection_count': 0,
                        'has_quantity': False,
                        'quantity': 0
                    }

            logger.info(f"Scanned {len(results)} ROIs with detector")
            return results

        except Exception as e:
            logger.error(f"✗ Scan screen ROI with detector: {e}")
            return {}

    def compare_with_csv(self, extracted_data: Dict[str, Any],
                        expected_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        So sánh dữ liệu trích xuất với expected data từ CSV.

        Args:
            extracted_data: Dữ liệu OCR từ ROI (có thể là Dict[str, str] hoặc Dict[str, Dict])
            expected_data: Dữ liệu mong đợi từ CSV

        Returns:
            Tuple[bool, str]: (is_match, message)
        """
        if not expected_data:
            return True, "No expected data"

        # Lọc ra các field có trong FESTIVALS_ROI_CONFIG (bỏ qua meta fields)
        roi_fields = set(FESTIVALS_ROI_CONFIG.keys())
        comparable_fields = {k: v for k, v in expected_data.items()
                           if k in roi_fields and v}  # Chỉ compare field có value

        if not comparable_fields:
            return True, "No comparable fields"

        matches = 0
        mismatches = []

        for field, expected_value in comparable_fields.items():
            if field not in extracted_data:
                mismatches.append(f"{field}:missing")
                continue

            # Handle detection result format (Dict) vs simple OCR format (str)
            if isinstance(extracted_data[field], dict):
                # Detection result format
                extracted_value = str(extracted_data[field].get('text', '')).strip()
            else:
                # Simple OCR format
                extracted_value = str(extracted_data[field]).strip()

            expected_value_str = str(expected_value).strip()

            # Normalize để so sánh (case-insensitive, remove spaces)
            extracted_normalized = extracted_value.lower().replace(' ', '')
            expected_normalized = expected_value_str.lower().replace(' ', '')

            if extracted_normalized == expected_normalized:
                matches += 1
            elif extracted_normalized in expected_normalized or expected_normalized in extracted_normalized:
                # Partial match cũng tính là OK
                matches += 1
                logger.debug(f"Partial match: {field} - '{extracted_value}' ~ '{expected_value_str}'")
            else:
                mismatches.append(f"{field}:'{extracted_value}'≠'{expected_value_str}'")

        total = len(comparable_fields)
        is_ok = matches == total

        if is_ok:
            message = f"✓ {matches}/{total} matched"
        else:
            message = f"✗ {matches}/{total} matched ({', '.join(mismatches[:3])})"

        return is_ok, message

    def compare_with_csv_enhanced(self, extracted_data: Dict[str, Dict[str, Any]],
                                 expected_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        So sánh dữ liệu với detection info (enhanced version).
        Kết hợp cả OCR text và detection để verify chính xác hơn.

        Args:
            extracted_data: Dữ liệu từ scan_screen_roi_with_detector (Dict[str, Dict])
            expected_data: Dữ liệu mong đợi từ CSV

        Returns:
            Tuple[bool, str, Dict]: (is_match, message, detailed_results)
        """
        if not expected_data:
            return True, "No expected data", {}

        # Lọc ra các field có trong FESTIVALS_ROI_CONFIG
        roi_fields = set[str](FESTIVALS_ROI_CONFIG.keys())
        comparable_fields = {k: v for k, v in expected_data.items()
                           if k in roi_fields and v}

        if not comparable_fields:
            return True, "No comparable fields", {}

        matches = 0
        mismatches = []
        detailed_results = {}

        for field, expected_value in comparable_fields.items():
            if field not in extracted_data:
                mismatches.append(f"{field}:missing")
                detailed_results[field] = {'status': 'missing', 'expected': expected_value}
                continue

            roi_result = extracted_data[field]
            extracted_text = roi_result.get('text', '').strip()
            detected = roi_result.get('detected', False)
            detection_count = roi_result.get('detection_count', 0)
            has_quantity = roi_result.get('has_quantity', False)
            quantity = roi_result.get('quantity', 0)

            expected_value_str = str(expected_value).strip()

            # Normalize
            extracted_normalized = extracted_text.lower().replace(' ', '')
            expected_normalized = expected_value_str.lower().replace(' ', '')

            # Compare logic
            text_match = False
            if extracted_normalized == expected_normalized:
                text_match = True
            elif extracted_normalized in expected_normalized or expected_normalized in extracted_normalized:
                text_match = True

            # Detailed result
            result_detail = {
                'status': 'match' if text_match else 'mismatch',
                'extracted_text': extracted_text,
                'expected': expected_value_str,
                'detected': detected,
                'detection_count': detection_count,
                'has_quantity': has_quantity,
                'quantity': quantity
            }
            detailed_results[field] = result_detail

            if text_match:
                matches += 1
                if detected:
                    logger.debug(f"✓ {field}: text='{extracted_text}', detected={detection_count} objects")
                else:
                    logger.debug(f"✓ {field}: text='{extracted_text}' (no detection)")
            else:
                mismatches.append(f"{field}:'{extracted_text}'≠'{expected_value_str}'")

        total = len(comparable_fields)
        is_ok = matches == total

        if is_ok:
            message = f"✓ {matches}/{total} matched"
        else:
            message = f"✗ {matches}/{total} matched ({', '.join(mismatches[:3])})"

        return is_ok, message, detailed_results

    def run_festival_stage(self, stage_data: Dict[str, Any], stage_idx: int,
                          use_detector: bool = False) -> bool:
        """
        Chạy automation cho 1 stage theo flow 1-16.

        Args:
            stage_data: Dữ liệu stage từ CSV/JSON
            stage_idx: Index của stage
            use_detector: Có sử dụng detector (YOLO/Template) không

        Returns:
            bool: True nếu pass, False nếu fail
        """
        logger.info(f"\n{'='*60}\nSTAGE {stage_idx}: {stage_data.get('フェス名', 'Unknown')}\n{'='*60}")

        rank = stage_data.get('推奨ランク', 'Unknown')
        folder_name = f"rank_{rank}_stage_{stage_idx}"
        search_text = stage_data.get('フェスランク', '')

        try:
            # Steps 1-2: Touch Festival và Event
            logger.info("Steps 1-2: Festival & Event")
            if not self.touch_template("tpl_festival.png") or not self.touch_template("tpl_event.png"):
                return False

            # Step 3: Snapshot before
            logger.info("Step 3: Snapshot before")
            if not self.snapshot_and_save(folder_name, "01_before.png"):
                return False

            # Step 4: Find and touch
            logger.info("Step 4: Find & touch")
            if not search_text or not self.find_and_touch(search_text):
                return False

            # Step 5: Snapshot after
            logger.info("Step 5: Snapshot after")
            screenshot_after = self.snapshot_and_save(folder_name, "02_after.png")
            if screenshot_after is None:
                return False

            # Step 6: ROI scan & compare (Pre-battle)
            logger.info("Step 6: ROI scan & compare (Pre-battle)")
            pre_battle_rois = FESTIVAL_CONFIG.get('pre_battle_rois', [
                'フェス名', 'フェスランク', '勝利点数', '推奨ランク',
                'Sランクボーダー', '初回クリア報酬', 'Sランク報酬'
            ])
            
            if use_detector and self.detector is not None:
                # Sử dụng detector + OCR
                extracted_before = self.scan_screen_roi_with_detector(screenshot_after, pre_battle_rois)
                is_ok_before, msg_before, details_before = self.compare_with_csv_enhanced(extracted_before, stage_data)
                logger.info(f"Pre-battle check (with detector): {msg_before}")
            else:
                # Chỉ OCR truyền thống
                extracted_before = self.scan_screen_roi(screenshot_after, pre_battle_rois)
                is_ok_before, msg_before = self.compare_with_csv(extracted_before, stage_data)
                logger.info(f"Pre-battle check: {msg_before}")

            # Steps 7-10: Battle flow
            logger.info("Steps 7-10: Battle flow")
            self.touch_template("tpl_challenge.png")
            self.touch_template("tpl_ok.png", optional=True)
            self.touch_template("tpl_allskip.png", optional=True)
            self.touch_template("tpl_ok.png", optional=True)
            time.sleep(2.0)

            # Step 11: Touch Result
            logger.info("Step 11: Touch Result")
            self.touch_template("tpl_result.png", optional=True)

            # Step 12: Snapshot result
            logger.info("Step 12: Snapshot result")
            screenshot_result = self.snapshot_and_save(folder_name, "03_result.png")

            # Step 13: ROI scan & compare (Post-battle)
            logger.info("Step 13: ROI scan & compare (Post-battle)")
            post_battle_rois = FESTIVAL_CONFIG.get('post_battle_rois', [
                '獲得ザックマネー', '獲得アイテム',
                '獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース'
            ])
            
            if use_detector and self.detector is not None:
                # Sử dụng detector + OCR
                extracted_after = self.scan_screen_roi_with_detector(screenshot_result, post_battle_rois)
                is_ok_after, msg_after, details_after = self.compare_with_csv_enhanced(extracted_after, stage_data)
                logger.info(f"Post-battle check (with detector): {msg_after}")
            else:
                # Chỉ OCR truyền thống
                extracted_after = self.scan_screen_roi(screenshot_result, post_battle_rois)
                is_ok_after, msg_after = self.compare_with_csv(extracted_after, stage_data)
                logger.info(f"Post-battle check: {msg_after}")

            # Steps 14-15: Close
            logger.info("Steps 14-15: Close")
            self.touch_template("tpl_ok.png", optional=True)
            self.touch_template("tpl_ok.png", optional=True)

            # Final result
            final = is_ok_before and is_ok_after
            logger.info(f"{'='*60}\n{'✓ OK' if final else '✗ NG'}: Stage {stage_idx}\n{'='*60}")
            return final

        except Exception as e:
            logger.error(f"✗ Stage {stage_idx}: {e}")
            return False

    def run_all_stages(self, data_path: str, output_path: Optional[str] = None,
                      use_detector: bool = False) -> bool:
        """
        Chạy automation cho tất cả stages.

        Args:
            data_path: Đường dẫn đến file CSV/JSON chứa test data
            output_path: Đường dẫn output result (None = auto-generate)
            use_detector: Có sử dụng detector (YOLO/Template) không

        Returns:
            bool: True nếu thành công
        """
        try:
            # Load data
            stages_data = load_json(data_path) if data_path.endswith('.json') else load_csv(data_path)
            if not stages_data:
                return False

            # Setup output
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detector_suffix = "_detector" if use_detector else ""
                output_path = os.path.join(self.results_dir, f"results_{timestamp}{detector_suffix}.csv")

            result_writer = ResultWriter(output_path)
            
            # Log mode
            mode = "Detector + OCR" if use_detector and self.detector else "OCR only"
            logger.info(f"Mode: {mode} | Stages: {len(stages_data)} | Output: {output_path}")

            # Process each stage
            for idx, stage_data in enumerate(stages_data, 1):
                test_case = stage_data.copy()
                test_case['test_case_id'] = idx

                is_ok = self.run_festival_stage(stage_data, idx, use_detector=use_detector)
                result_writer.add_result(test_case,
                                       ResultWriter.RESULT_OK if is_ok else ResultWriter.RESULT_NG,
                                       error_message=None if is_ok else "Verification failed")
                time.sleep(1.0)

            # Save results
            result_writer.write()
            result_writer.print_summary()
            return True

        except Exception as e:
            logger.error(f"✗ All stages: {e}")
            return False

    def run(self, data_path: str, use_detector: bool = False) -> bool:
        """
        Entry point chính.

        Args:
            data_path: Đường dẫn đến file CSV/JSON chứa test data
            use_detector: Có sử dụng detector (YOLO/Template) không

        Returns:
            bool: True nếu thành công
        """
        logger.info("="*70 + "\nFESTIVAL AUTOMATION START\n" + "="*70)

        if not self.agent.is_device_connected():
            logger.error("✗ Device not connected")
            return False

        success = self.run_all_stages(data_path, use_detector=use_detector)
        logger.info("="*70 + f"\n{'✓ COMPLETED' if success else '✗ FAILED'}\n" + "="*70)
        return success

