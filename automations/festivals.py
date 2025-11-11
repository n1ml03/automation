"""
Festival Automation

Standard flow:
1. touch(Template("tpl_festival.png"))
2. touch(Template("tpl_event.png"))
3. snapshot -> save to folder (rank_E_stage_1/01_before_touch.png)
4. find_and_touch (OCR -> find text -> touch)
5. snapshot -> save to folder (rank_E_stage_1/02_after_touch.png)
6. ROI scan -> compare CSV -> record OK/NG
7. touch(Template("tpl_challenge.png"))
8. touch(Template("tpl_ok.png"))
9. touch(Template("tpl_allskip.png"))
10. touch(Template("tpl_ok.png"))
11. touch(Template("tpl_result.png"))
12. snapshot -> save to folder (rank_E_stage_1/03_result.png)
13. ROI scan -> compare CSV -> record OK/NG
14. touch(Template("tpl_ok.png")) if exists
15. touch(Template("tpl_ok.png")) if exists
16. Repeat
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from airtest.core.api import sleep

from core.base import BaseAutomation
from core.agent import Agent
from core.utils import get_logger
from core.data import ResultWriter, load_data
from core.config import (
    FESTIVALS_ROI_CONFIG, get_festivals_roi_config,
    FESTIVAL_CONFIG, get_festival_config, merge_config
)
from core.detector import YOLODetector, TemplateMatcher, YOLO_AVAILABLE

logger = get_logger(__name__)


class FestivalAutomation(BaseAutomation):
    """Festival automation - keep only essential steps."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        # Merge config: base config from FESTIVAL_CONFIG + custom config
        base_config = get_festival_config()
        cfg = merge_config(base_config, config) if config else base_config

        # Initialize base class with agent, config, and ROI config
        super().__init__(agent, cfg, FESTIVALS_ROI_CONFIG)

        # Initialize detector (YOLO or Template Matching)
        self.detector = None
        self.use_detector = cfg.get('use_detector')

        if self.use_detector:
            detector_type = cfg.get('detector_type', 'template')  # 'yolo', 'template', 'auto'

            if detector_type == 'auto':
                # Auto-select: prefer YOLO, fallback to Template
                if YOLO_AVAILABLE:
                    try:
                        yolo_config = cfg.get('yolo_config', {})
                        self.detector = YOLODetector(
                            agent=agent,
                            model_path=yolo_config.get('model_path', 'yolo11n.pt'),
                            confidence=yolo_config.get('confidence', 0.25),
                            device=yolo_config.get('device', 'cpu')
                        )
                        logger.info("Using YOLO Detector")
                    except Exception as e:
                        logger.warning(f"YOLO init failed: {e}, fallback to Template")
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
                    logger.info("Using Template Matcher")

            elif detector_type == 'yolo':
                yolo_config = cfg.get('yolo_config', {})
                self.detector = YOLODetector(
                    agent=agent,
                    model_path=yolo_config.get('model_path', 'yolo11n.pt'),
                    confidence=yolo_config.get('confidence', 0.25),
                    device=yolo_config.get('device', 'cpu')
                )
                logger.info("Using YOLO Detector")

            elif detector_type == 'template':
                template_config = cfg.get('template_config', {})
                self.detector = TemplateMatcher(
                    templates_dir=template_config.get('templates_dir', self.templates_path),
                    threshold=template_config.get('threshold', 0.85)
                )
                logger.info("Using Template Matcher")

        logger.info("FestivalAutomation initialized")

    def detect_and_ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> Dict[str, Any]:
        """
        Detect objects in ROI and OCR to get text/quantity.
        Combine detector (YOLO/Template) with OCR for better accuracy.

        Args:
            roi_name: ROI name in FESTIVALS_ROI_CONFIG
            screenshot: Screenshot to scan, None = take new

        Returns:
            Dict[str, Any]:
            {
                'roi_name': str,
                'text': str,  # Text from OCR
                'detected': bool,  # Object detected
                'detections': List[Dict],  # Detected objects list
                'detection_count': int,  # Number of detected objects
                'has_quantity': bool,  # Has quantity (from detector)
                'quantity': int,  # Quantity if available (from detector)
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
            # Get ROI config
            roi_config = get_festivals_roi_config(roi_name)
            coords = roi_config['coords']  # [x1, x2, y1, y2]
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # Step 1: Traditional OCR (always runs)
            if screenshot is None:
                # Use agent.ocr() with region - more efficient
                ocr_result = self.agent.ocr(region)
                if ocr_result is None:
                    logger.warning(f"✗ Cannot OCR ROI '{roi_name}'")
                    return result
                text = self._clean_ocr_text(ocr_result.get('text', ''))
                result['text'] = text
                logger.debug(f"ROI '{roi_name}' OCR: '{text}'")
                
                # Get screenshot for detector if needed
                screenshot = self.agent.snapshot()
                if screenshot is None:
                    logger.warning(f"✗ Cannot get screenshot for detector")
                    return result
            else:
                # Use existing screenshot
                if self.agent.ocr_engine is not None:
                    ocr_result = self.agent.ocr(region) if screenshot is None else None
                    if ocr_result is None:
                        roi_image = screenshot[y1:y2, x1:x2]
                        if roi_image is None or roi_image.size == 0:
                            logger.warning(f"✗ ROI '{roi_name}': Invalid crop")
                            return result
                        ocr_result = self.agent.ocr_engine.recognize(roi_image)
                    text = self._clean_ocr_text(ocr_result.get('text', ''))
                    result['text'] = text
                    logger.debug(f"ROI '{roi_name}' OCR: '{text}'")

            # Crop ROI for detector
            roi_image = screenshot[y1:y2, x1:x2]
            if roi_image is None or roi_image.size == 0:
                logger.warning(f"✗ ROI '{roi_name}': Invalid crop for detector")
                return result

            # Step 2: Detection (if detector available)
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

                        # Save quantity from first detection
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

    def scan_screen_roi_with_detector(self, screenshot: Optional[Any] = None,
                                     roi_names: Optional[List[str]] = None,
                                     use_detector: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Scan screen with detector (YOLO/Template) + OCR.
        Enhanced version of scan_screen_roi with detection info.

        Args:
            screenshot: Screenshot to scan, None = take new
            roi_names: List of ROI names to scan, None = scan all
            use_detector: Use detector (True = detect + OCR, False = OCR only)

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with ROI name as key, detection result as value
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
            # Get screenshot if not provided
            if screenshot is None:
                screenshot = self.agent.snapshot()
                if screenshot is None:
                    logger.error("Cannot get screenshot")
                    return {}

            # Determine ROI list to scan
            if roi_names is None:
                roi_names = list(FESTIVALS_ROI_CONFIG.keys())

            # Scan each ROI with detector
            results = {}
            for roi_name in roi_names:
                try:
                    if use_detector and self.detector is not None:
                        # Use detector + OCR
                        result = self.detect_and_ocr_roi(roi_name, screenshot)
                        results[roi_name] = result
                    else:
                        # Fallback: OCR only
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

    def compare_results(self, extracted_data: Dict[str, Any],
                       expected_data: Dict[str, Any],
                       return_details: bool = True) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Compare OCR/Detector data with expected CSV data.
        Supports both simple OCR format (Dict[str, str]) and detector format (Dict[str, Dict]).

        Args:
            extracted_data: Data from OCR or Detector
                - Simple format: Dict[str, str] from scan_screen_roi()
                - Detector format: Dict[str, Dict] from scan_screen_roi_with_detector()
            expected_data: Expected data from CSV
            return_details: Return detailed results (default: True)

        Returns:
            Tuple[bool, str, Optional[Dict]]: (is_match, message, detailed_results)
                - is_match: True if match, False if not
                - message: Summary message
                - detailed_results: Field details (None if return_details=False)
        """
        if not expected_data:
            return True, "No expected data", None if not return_details else {}

        # Filter fields in FESTIVALS_ROI_CONFIG (ignore meta fields)
        roi_fields = set(FESTIVALS_ROI_CONFIG.keys())
        comparable_fields = {k: v for k, v in expected_data.items()
                           if k in roi_fields and v}  # Only compare fields with values

        if not comparable_fields:
            return True, "No comparable fields", None if not return_details else {}

        matches = 0
        mismatches = []
        detailed_results: Dict[str, Any] = {}

        for field, expected_value in comparable_fields.items():
            if field not in extracted_data:
                mismatches.append(f"{field}:missing")
                if return_details:
                    detailed_results[field] = {'status': 'missing', 'expected': expected_value}
                continue

            # Handle both simple OCR format (str) and detector format (dict)
            field_data = extracted_data[field]
            if isinstance(field_data, dict):
                # Detector format: extract details
                extracted_text = field_data.get('text', '').strip()
                detected = field_data.get('detected', False)
                detection_count = field_data.get('detection_count', 0)
                has_quantity = field_data.get('has_quantity', False)
                quantity = field_data.get('quantity', 0)
            else:
                # Simple OCR format: just text
                extracted_text = str(field_data).strip()
                detected = False
                detection_count = 0
                has_quantity = False
                quantity = 0

            expected_value_str = str(expected_value).strip()

            # Normalize for comparison (case-insensitive, remove spaces)
            extracted_normalized = extracted_text.lower().replace(' ', '')
            expected_normalized = expected_value_str.lower().replace(' ', '')

            # Compare logic
            text_match = False
            if extracted_normalized == expected_normalized:
                text_match = True
            elif extracted_normalized in expected_normalized or expected_normalized in extracted_normalized:
                # Partial match also counts as OK
                text_match = True
                logger.debug(f"Partial match: {field} - '{extracted_text}' ~ '{expected_value_str}'")

            # Store detailed results if requested
            if return_details:
                detailed_results[field] = {
                    'status': 'match' if text_match else 'mismatch',
                    'extracted_text': extracted_text,
                    'expected': expected_value_str,
                    'detected': detected,
                    'detection_count': detection_count,
                    'has_quantity': has_quantity,
                    'quantity': quantity
                }

            if text_match:
                matches += 1
                if detected:
                    logger.debug(f"✓ {field}: text='{extracted_text}', detected={detection_count} objects")
                else:
                    logger.debug(f"✓ {field}: text='{extracted_text}'")
            else:
                mismatches.append(f"{field}:'{extracted_text}'≠'{expected_value_str}'")

        total = len(comparable_fields)
        is_ok = matches == total

        if is_ok:
            message = f"✓ {matches}/{total} matched"
        else:
            message = f"✗ {matches}/{total} matched ({', '.join(mismatches[:3])})"

        return is_ok, message, detailed_results if return_details else None

    def run_festival_stage(self, stage_data: Dict[str, Any], stage_idx: int,
                          use_detector: bool = False) -> bool:
        """
        Run automation for 1 stage following flow 1-16.

        Args:
            stage_data: Stage data from CSV/JSON
            stage_idx: Stage index
            use_detector: Use detector (YOLO/Template)

        Returns:
            bool: True if pass, False if fail
        """
        logger.info(f"\n{'='*60}\nSTAGE {stage_idx}: {stage_data.get('フェス名', 'Unknown')}\n{'='*60}")

        rank = stage_data.get('推奨ランク', 'Unknown')
        folder_name = f"rank_{rank}_stage_{stage_idx}"
        search_text = stage_data.get('フェスランク', '')

        try:
            # Step 1: Touch Festival
            logger.info("Step 1: Touch Festival")
            if not self.touch_template("tpl_festival.png"):
                logger.error("Step 1: Failed to touch festival button")
                return False

            # Wait for festival menu to load
            sleep(0.5)

            # Step 2: Touch Event
            logger.info("Step 2: Touch Event")
            if not self.touch_template("tpl_event.png"):
                logger.error("Step 2: Failed to touch event button")
                return False

            # Wait for event screen to load
            sleep(0.5)

            # Step 3: Snapshot before touch
            logger.info("Step 3: Snapshot before touch")
            screenshot_before = self.snapshot_and_save(folder_name, "01_before_touch.png")
            if screenshot_before is None:
                logger.error("Step 3: Failed to take before touch snapshot")
                return False

            # Step 4: Find and touch (OCR search)
            logger.info("Step 4: Find & touch (OCR)")
            if not search_text:
                logger.error("Step 4: No search text provided")
                return False

            if not self.find_and_touch(search_text):
                logger.error(f"Step 4: Failed to find and touch '{search_text}'")
                return False

            # Wait for screen transition after touch
            sleep(0.5)

            # Step 5: Snapshot after touch
            logger.info("Step 5: Snapshot after touch")
            screenshot_after = self.snapshot_and_save(folder_name, "02_after_touch.png")
            if screenshot_after is None:
                logger.error("Step 5: Failed to take after touch snapshot")
                return False

            # Step 6: ROI scan & compare (Pre-battle verification)
            logger.info("Step 6: ROI scan & compare (Pre-battle)")
            pre_battle_rois = FESTIVAL_CONFIG.get('pre_battle_rois', [
                'フェス名', 'フェスランク', '勝利点数', '推奨ランク',
                'Sランクボーダー', '初回クリア報酬', 'Sランク報酬'
            ])

            if use_detector and self.detector is not None:
                # Use detector + OCR
                extracted_before = self.scan_screen_roi_with_detector(screenshot_after, pre_battle_rois)
                is_ok_before, msg_before, details_before = self.compare_results(extracted_before, stage_data)
                logger.info(f"Pre-battle verification (with detector): {msg_before}")
            else:
                # Traditional OCR only
                extracted_before = self.scan_screen_roi(screenshot_after, pre_battle_rois)
                is_ok_before, msg_before, _ = self.compare_results(extracted_before, stage_data, return_details=False)
                logger.info(f"Pre-battle verification: {msg_before}")

            # Pre-battle verification must pass
            if not is_ok_before:
                logger.error(f"Step 6: Pre-battle verification failed: {msg_before}")
                return False

            # Step 7: Touch Challenge
            logger.info("Step 7: Touch Challenge")
            if not self.touch_template("tpl_challenge.png"):
                logger.warning("Step 7: Failed to touch challenge button")
                return False

            # Step 8: Touch OK (confirmation dialog)
            logger.info("Step 8: Touch OK (confirmation)")
            self.touch_template("tpl_ok.png", optional=True)

            # Step 9: Touch All Skip
            logger.info("Step 9: Touch All Skip")
            if not self.touch_template("tpl_allskip.png"):
                logger.warning("Step 9: Failed to touch all skip button")
                return False

            # Step 10: Touch OK (after skip)
            logger.info("Step 10: Touch OK (after skip)")
            self.touch_template("tpl_ok.png", optional=True)

            # Wait for battle to complete
            logger.info("Waiting for battle completion...")
            sleep(2.0)

            # Step 11: Touch Result
            logger.info("Step 11: Touch Result")
            if not self.touch_template("tpl_result.png"):
                logger.warning("Step 11: Failed to touch result button")
                return False

            # Step 12: Snapshot result
            logger.info("Step 12: Snapshot result")
            screenshot_result = self.snapshot_and_save(folder_name, "03_result.png")
            if screenshot_result is None:
                return False

            # Step 13: ROI scan & compare (Post-battle)
            logger.info("Step 13: ROI scan & compare (Post-battle)")
            post_battle_rois = FESTIVAL_CONFIG.get('post_battle_rois', [
                '獲得ザックマネー', '獲得アイテム',
                '獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース'
            ])

            if use_detector and self.detector is not None:
                # Use detector + OCR
                extracted_after = self.scan_screen_roi_with_detector(screenshot_result, post_battle_rois)
                is_ok_after, msg_after, details_after = self.compare_results(extracted_after, stage_data)
                logger.info(f"Post-battle check (with detector): {msg_after}")
            else:
                # Traditional OCR only
                extracted_after = self.scan_screen_roi(screenshot_result, post_battle_rois)
                is_ok_after, msg_after, _ = self.compare_results(extracted_after, stage_data, return_details=False)
                logger.info(f"Post-battle check: {msg_after}")

            # Step 14: Touch OK to close result (first OK)
            logger.info("Step 14: Touch OK (close result - first)")
            self.touch_template("tpl_ok.png", optional=True)

            # Step 15: Touch OK to close result (second OK if needed)
            logger.info("Step 15: Touch OK (close result - second)")
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
        Run automation for all stages.

        Args:
            data_path: Path to CSV/JSON file with test data
            output_path: Output result path (None = auto-generate)
            use_detector: Use detector (YOLO/Template)

        Returns:
            bool: True if successful
        """
        try:
            # Load data
            stages_data = load_data(data_path)
            if not stages_data:
                return False

            # Setup output
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detector_suffix = "_detector" if use_detector else ""
                output_path = f"{self.results_dir}/results_{timestamp}{detector_suffix}.csv"

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
                sleep(1.0)

            # Save results
            result_writer.write()
            result_writer.print_summary()
            return True

        except Exception as e:
            logger.error(f"✗ All stages: {e}")
            return False

    def run(self, data_path: str, use_detector: bool = False) -> bool:
        """
        Main entry point.

        Args:
            data_path: Path to CSV/JSON file with test data
            use_detector: Use detector (YOLO/Template)

        Returns:
            bool: True if successful
        """
        logger.info("="*70 + "\nFESTIVAL AUTOMATION START\n" + "="*70)

        if not self.agent.is_device_connected():
            logger.error("✗ Device not connected")
            return False

        success = self.run_all_stages(data_path, use_detector=use_detector)
        logger.info("="*70 + f"\n{'✓ COMPLETED' if success else '✗ FAILED'}\n" + "="*70)
        return success
