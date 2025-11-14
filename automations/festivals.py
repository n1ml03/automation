"""
Festival Automation

Standard flow:
1. touch(Template("tpl_festival.png"))
2. touch(Template("tpl_event.png"))
3. snapshot -> save to folder (rank_E_stage_1/01_before_touch.png)
4. find_and_touch_in_roi('フェス名', stage_text) - OCR in フェス名 ROI -> find text -> touch
4.5. find_and_touch_in_roi('フェスランク', rank_text) - OCR in フェスランク ROI -> find text -> touch
5. snapshot -> save to folder (rank_E_stage_1/02_after_touch.png)
6. ROI scan -> compare CSV -> record OK/NG (Pre-battle verification)
7. touch(Template("tpl_challenge.png"))
8. touch(Template("tpl_ok.png")) - confirmation dialog
9. touch(Template("tpl_allskip.png"))
10. touch(Template("tpl_ok.png")) - after skip
11. touch(Template("tpl_result.png"))
12. snapshot -> save to folder (rank_E_stage_1/03_result.png)
13. ROI scan -> compare CSV -> record OK/NG (Post-battle verification)
14. touch(Template("tpl_ok.png")) if exists - close result (first)
15. touch(Template("tpl_ok.png")) if exists - close result (second)
16. Repeat

Key Features:
- ROI-based text detection: Uses find_and_touch_in_roi() to search text within specific 
  screen regions, improving accuracy and avoiding false positives
- Auto-retry: Each step automatically retries on failure
- Cancellation support: Can be interrupted at any time
- Resume capability: Can continue from where it left off
- Detector support: Optional YOLO/Template detection for enhanced verification
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from airtest.core.api import sleep

from core.base import BaseAutomation, CancellationError
from core.agent import Agent
from core.utils import get_logger
from core.data import ResultWriter, load_data
from core.config import (
    FESTIVALS_ROI_CONFIG, get_festivals_roi_config,
    FESTIVAL_CONFIG, get_festival_config, merge_config
)
from core.detector import YOLODetector, TemplateMatcher, YOLO_AVAILABLE, OCRTextProcessor

logger = get_logger(__name__)


class FestivalAutomation(BaseAutomation):
    """Festival automation - keep only essential steps."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None):
        # Merge config: base config from FESTIVAL_CONFIG + custom config
        base_config = get_festival_config()
        cfg = merge_config(base_config, config) if config else base_config

        # Initialize base class with agent, config, ROI config, and cancellation event
        super().__init__(agent, cfg, FESTIVALS_ROI_CONFIG, cancel_event=cancel_event)

        # Store config for later use
        self.config = cfg

        # Initialize detector (YOLO or Template Matching) using factory pattern
        self.detector = None
        self.use_detector = cfg.get('use_detector')

        if self.use_detector:
            self.detector = self._create_detector(cfg, agent)

        logger.info("FestivalAutomation initialized")

    def _create_detector(self, cfg: Dict[str, Any], agent: Agent) -> Optional[Any]:
        """
        Factory method for creating detectors (YOLO or Template Matching).
        
        Args:
            cfg: Configuration dictionary
            agent: Agent instance
            
        Returns:
            Detector instance or None if creation fails
        """
        detector_type = cfg.get('detector_type', 'template')  # 'yolo', 'template', 'auto'
        
        def create_yolo():
            """Create YOLO detector."""
            yolo_config = cfg.get('yolo_config', {})
            return YOLODetector(
                agent=agent,
                model_path=yolo_config.get('model_path', 'yolo11n.pt'),
                confidence=yolo_config.get('confidence', 0.25),
                device=yolo_config.get('device', 'cpu')
            )
        
        def create_template():
            """Create Template Matcher."""
            template_config = cfg.get('template_config', {})
            return TemplateMatcher(
                templates_dir=template_config.get('templates_dir', self.templates_path),
                threshold=template_config.get('threshold', 0.85)
            )
        
        if detector_type == 'auto':
            # Auto-select: prefer YOLO, fallback to Template
            if YOLO_AVAILABLE:
                try:
                    detector = create_yolo()
                    logger.info("Using YOLO Detector")
                    return detector
                except Exception as e:
                    logger.warning(f"YOLO init failed: {e}, fallback to Template")
            detector = create_template()
            logger.info("Using Template Matcher")
            return detector
        
        elif detector_type == 'yolo':
            try:
                detector = create_yolo()
                logger.info("Using YOLO Detector")
                return detector
            except Exception as e:
                logger.error(f"YOLO init failed: {e}")
                return None
        
        elif detector_type == 'template':
            detector = create_template()
            logger.info("Using Template Matcher")
            return detector
        
        return None

    def detect_roi(self, roi_name: str, screenshot: Optional[Any] = None,
                   roi_image: Optional[Any] = None) -> Dict[str, Any]:
        """
        Detect objects in ROI using detector (YOLO/Template).
        Focused function that only performs detection without OCR.
        
        Note: YOLO detector already includes OCR for quantity extraction internally.

        Args:
            roi_name: ROI name in FESTIVALS_ROI_CONFIG
            screenshot: Screenshot to scan, None = take new
            roi_image: Pre-cropped ROI image (optimization to avoid double crop)

        Returns:
            Dict[str, Any]:
            {
                'roi_name': str,
                'detected': bool,
                'detections': List[Dict],  # Each detection has: item, quantity, confidence, etc.
                'detection_count': int
            }
        """
        result = {
            'roi_name': roi_name,
            'detected': False,
            'detections': [],
            'detection_count': 0
        }

        try:
            if self.detector is None:
                logger.warning(f"Detector not available for ROI '{roi_name}'")
                return result

            # Use pre-cropped ROI if provided, otherwise crop from screenshot
            if roi_image is None:
                # Get screenshot using helper
                screenshot = self.get_screenshot(screenshot)
                if screenshot is None:
                    return result

                # Crop ROI using helper
                roi_image = self.crop_roi(screenshot, roi_name)
                if roi_image is None:
                    return result

            # Detect objects in ROI
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

            logger.info(f"✓ ROI '{roi_name}': detected={result['detected']} ({result['detection_count']} objects)")
            return result

        except Exception as e:
            logger.error(f"✗ Detect ROI '{roi_name}': {e}")
            return result

    def scan_rois_detector(self, screenshot: Optional[Any] = None,
                          roi_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Scan multiple ROIs using detector.
        Simplified version focused only on detection.

        Args:
            screenshot: Screenshot to scan, None = take new
            roi_names: List of ROI names to scan, None = scan all

        Returns:
            Dict[str, Dict[str, Any]]: Detection results for each ROI
            {
                'roi_name': {
                    'detected': bool,
                    'detections': List[Dict],
                    'detection_count': int
                }
            }
        """
        try:
            if self.detector is None:
                logger.error("Detector not available")
                return {}

            # Get screenshot using helper
            screenshot = self.get_screenshot(screenshot)
            if screenshot is None:
                return {}

            # Determine ROI list to scan
            if roi_names is None:
                roi_names = list(FESTIVALS_ROI_CONFIG.keys())

            # Scan each ROI with detector
            results = {}
            for roi_name in roi_names:
                result = self.detect_roi(roi_name, screenshot)
                results[roi_name] = result

            logger.info(f"Scanned {len(results)} ROIs with detector")
            return results

        except Exception as e:
            logger.error(f"✗ Scan ROIs with detector: {e}")
            return {}

    def scan_rois_combined(self, screenshot: Optional[Any] = None,
                          roi_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Scan multiple ROIs with both OCR and detector.
        Combines OCR text with detection results for comprehensive verification.
        
        Optimized: Crops each ROI only ONCE and uses for both OCR and detection.

        Args:
            screenshot: Screenshot to scan, None = take new
            roi_names: List of ROI names to scan, None = scan all

        Returns:
            Dict[str, Dict[str, Any]]: Combined results for each ROI
            {
                'roi_name': {
                    'text': str,           # From OCR
                    'detected': bool,      # From detector
                    'detections': List[Dict],
                    'detection_count': int
                }
            }
        """
        try:
            # Get screenshot using helper
            screenshot = self.get_screenshot(screenshot)
            if screenshot is None:
                return {}

            # Determine ROI list to scan
            if roi_names is None:
                roi_names = list(FESTIVALS_ROI_CONFIG.keys())

            # Scan each ROI - OPTIMIZED: crop only once per ROI
            results = {}
            for roi_name in roi_names:
                # Crop ROI once using helper (from base.py)
                roi_image = self.crop_roi(screenshot, roi_name)
                if roi_image is None:
                    results[roi_name] = {
                        'roi_name': roi_name,
                        'text': '',
                        'detected': False,
                        'detections': [],
                        'detection_count': 0
                    }
                    continue

                # Get ROI config for OCR
                roi_config = self.get_roi_config(roi_name)
                if roi_config is None:
                    continue

                # OCR for text - use roi_image directly
                text = ''
                if self.agent.ocr_engine is not None:
                    try:
                        ocr_result = self.agent.ocr_engine.recognize(roi_image)
                        if ocr_result:
                            text = self._clean_ocr_text(ocr_result.get('text', ''))
                    except Exception as e:
                        logger.debug(f"OCR failed for '{roi_name}': {e}")
                
                # Detection - pass pre-cropped roi_image to avoid double crop
                if self.detector is not None:
                    detection_result = self.detect_roi(roi_name, roi_image=roi_image)
                    results[roi_name] = {
                        'roi_name': roi_name,
                        'text': text,
                        'detected': detection_result['detected'],
                        'detections': detection_result['detections'],
                        'detection_count': detection_result['detection_count']
                    }
                else:
                    results[roi_name] = {
                        'roi_name': roi_name,
                        'text': text,
                        'detected': False,
                        'detections': [],
                        'detection_count': 0
                    }

            logger.info(f"Scanned {len(results)} ROIs (combined OCR + detector, optimized)")
            return results

        except Exception as e:
            logger.error(f"✗ Scan ROIs combined: {e}")
            return {}

    def compare_results(self, extracted_data: Dict[str, Any],
                       expected_data: Dict[str, Any],
                       return_details: bool = True) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Compare OCR/Detector data with expected CSV data using OCRTextProcessor.
        Supports both simple OCR format (Dict[str, str]) and combined format (Dict[str, Dict]).

        Args:
            extracted_data: Data from OCR or Detector
                - Simple format: Dict[str, str] from scan_screen_roi()
                - Combined format: Dict[str, Dict] from scan_rois_combined()
            expected_data: Expected data from CSV
            return_details: Return detailed results (default: True)

        Returns:
            Tuple[bool, str, Optional[Dict]]: (is_match, message, detailed_results)
                - is_match: True if match, False if not
                - message: Summary message
                - detailed_results: Field details with extracted values (None if return_details=False)
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

            # Handle both simple OCR format (str) and combined format (dict)
            field_data = extracted_data[field]
            if isinstance(field_data, dict):
                # Combined format: extract details
                extracted_text = field_data.get('text', '').strip()
                detected = field_data.get('detected', False)
                detection_count = field_data.get('detection_count', 0)
                detections = field_data.get('detections', [])
                
                # Extract quantity from first detection if available
                has_quantity = False
                quantity = 0
                if detections and len(detections) > 0:
                    first_detection = detections[0]
                    quantity = first_detection.get('quantity', 0)
                    has_quantity = quantity > 0
            else:
                # Simple OCR format: just text
                extracted_text = str(field_data).strip()
                detected = False
                detection_count = 0
                has_quantity = False
                quantity = 0

            # Use OCRTextProcessor for field-specific validation and extraction (returns dataclass)
            validation_result = OCRTextProcessor.validate_field(field, extracted_text, expected_value)
            text_match = validation_result.status == 'match'
            
            # Store detailed results if requested
            if return_details:
                detailed_results[field] = {
                    'status': validation_result.status,
                    'extracted_text': extracted_text,
                    'extracted_value': validation_result.extracted,
                    'expected': validation_result.expected,
                    'detected': detected,
                    'detection_count': detection_count,
                    'has_quantity': has_quantity,
                    'quantity': quantity,
                    'message': validation_result.message,
                    'confidence': validation_result.confidence
                }
            
            if text_match:
                matches += 1
                logger.debug(f"✓ {field}: {validation_result.message}")
            else:
                mismatch_msg = f"{field}:{validation_result.extracted}≠{validation_result.expected}"
                mismatches.append(mismatch_msg)
                logger.debug(f"✗ {field}: {validation_result.message}")

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
        Each step will retry until success before moving to next step.

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
        stage_text = stage_data.get('フェス名', '')
        rank_text = stage_data.get('フェスランク', '')
        max_retries = self.config.get('max_step_retries', 5)
        retry_delay = self.config.get('retry_delay', 1.0)
        festivals_rois = FESTIVAL_CONFIG.get('festivals_rois', ['フェス名', 'フェスランク'])

        try:
            # Step 1: Touch Festival (with retry)
            logger.info("Step 1: Touch Festival")
            if not self.retry_with_cancellation(lambda: self.touch_template("tpl_festival.png"), max_retries, retry_delay, "Step 1: Touch Festival"):
                return False
            logger.info("✓ Step 1: Successfully touched festival button")
            # Wait for festival menu to load
            sleep(0.5)

            # Step 2: Touch Event (with retry)
            logger.info("Step 2: Touch Event")
            if not self.retry_with_cancellation(lambda: self.touch_template("tpl_event.png"), max_retries, retry_delay, "Step 2: Touch Event"):
                return False
            logger.info("✓ Step 2: Successfully touched event button")
            # Wait for event screen to load
            sleep(0.5)

            # Step 3: Snapshot before touch (with retry)
            logger.info("Step 3: Snapshot before touch")
            screenshot_before = self.retry_with_cancellation(lambda: self.snapshot_and_save(folder_name, "01_before_touch.png"), max_retries, retry_delay, "Step 3: Snapshot before touch")
            if screenshot_before is None:
                return False
            logger.info("✓ Step 3: Successfully captured before touch snapshot")

            # Step 4: Find and touch stage name in フェス名 ROI
            logger.info(f"Step 4: Find & touch '{stage_text}' in ROI 'フェス名'")
            if not stage_text:
                logger.error("Step 4: No stage text (フェス名) provided")
                return False
            if not self.retry_with_cancellation(
                lambda: self.find_and_touch_in_roi('フェス名', stage_text), 
                max_retries, retry_delay, 
                f"Step 4: Find & touch '{stage_text}' in ROI 'フェス名'"
            ):
                return False
            logger.info(f"✓ Step 4: Successfully found and touched '{stage_text}'")
            # Wait for screen transition after touch
            sleep(0.5)
            
            # Step 4.5: Find and touch rank in フェスランク ROI
            logger.info(f"Step 4.5: Find & touch '{rank_text}' in ROI 'フェスランク'")
            if not rank_text:
                logger.error("Step 4.5: No rank text (フェスランク) provided")
                return False
            if not self.retry_with_cancellation(
                lambda: self.find_and_touch_in_roi('フェスランク', rank_text), 
                max_retries, retry_delay,
                f"Step 4.5: Find & touch '{rank_text}' in ROI 'フェスランク'"
            ):
                return False
            logger.info(f"✓ Step 4.5: Successfully found and touched '{rank_text}'")
            # Wait for screen transition after touch
            sleep(0.5)

            # Step 5: Snapshot after touch (with retry)
            logger.info("Step 5: Snapshot after touch")
            screenshot_after = self.retry_with_cancellation(lambda: self.snapshot_and_save(folder_name, "02_after_touch.png"), max_retries, retry_delay, "Step 5: Snapshot after touch")
            if screenshot_after is None:
                return False
            logger.info("✓ Step 5: Successfully captured after touch snapshot")

            # Step 6: ROI scan & compare (Pre-battle verification with retry)
            logger.info("Step 6: ROI scan & compare (Pre-battle)")
            pre_battle_rois = FESTIVAL_CONFIG.get('pre_battle_rois', ['勝利点数', '推奨ランク', 'Sランクボーダー', '初回クリア報酬', 'Sランク報酬'])
            retry_count = 0
            is_ok_before = False
            while retry_count < max_retries:
                self.check_cancelled("Step 6: Pre-battle verification")
                if use_detector and self.detector is not None:
                    extracted = self.scan_rois_combined(screenshot_after, pre_battle_rois)
                    is_ok_before, msg, _ = self.compare_results(extracted, stage_data)
                    logger.info(f"Pre-battle verification (with detector): {msg}")
                else:
                    extracted = self.scan_screen_roi(screenshot_after, pre_battle_rois)
                    is_ok_before, msg, _ = self.compare_results(extracted, stage_data, return_details=False)
                    logger.info(f"Pre-battle verification: {msg}")
                if is_ok_before:
                    logger.info("✓ Step 6: Pre-battle verification passed")
                    break
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Step 6: Verification attempt {retry_count}/{max_retries} failed, retaking screenshot...")
                    screenshot_after = self.snapshot_and_save(folder_name, f"02_after_touch_retry{retry_count}.png")
                    sleep(retry_delay)
            else:
                logger.error(f"Step 6: Pre-battle verification failed after {max_retries} attempts")
                return False

            # Step 7: Touch Challenge (with retry)
            logger.info("Step 7: Touch Challenge")
            if not self.retry_with_cancellation(lambda: self.touch_template("tpl_challenge.png"), max_retries, retry_delay, "Step 7: Touch Challenge"):
                return False
            logger.info("✓ Step 7: Successfully touched challenge button")
            sleep(0.5)

            # Step 8: Touch OK (confirmation dialog - retry until found or max attempts)
            logger.info("Step 8: Touch OK (confirmation)")
            self.retry_with_cancellation(lambda: self.touch_template("tpl_ok.png", optional=True), max_retries, 0.3, "Step 8: Touch OK (confirmation)")
            sleep(0.5)

            # Step 9: Touch All Skip (with retry)
            logger.info("Step 9: Touch All Skip")
            if not self.retry_with_cancellation(lambda: self.touch_template("tpl_allskip.png"), max_retries, retry_delay, "Step 9: Touch All Skip"):
                return False
            logger.info("✓ Step 9: Successfully touched all skip button")
            sleep(0.5)

            # Step 10: Touch OK (after skip - retry until found)
            logger.info("Step 10: Touch OK (after skip)")
            self.retry_with_cancellation(lambda: self.touch_template("tpl_ok.png", optional=True), max_retries, 0.3, "Step 10: Touch OK (after skip)")
            # Wait for battle to complete
            logger.info("Waiting for battle completion...")
            sleep(2.0)

            # Step 11: Touch Result (with retry)
            logger.info("Step 11: Touch Result")
            if not self.retry_with_cancellation(lambda: self.touch_template("tpl_result.png"), max_retries, retry_delay, "Step 11: Touch Result"):
                return False
            logger.info("✓ Step 11: Successfully touched result button")
            sleep(0.5)

            # Step 12: Snapshot result (with retry)
            logger.info("Step 12: Snapshot result")
            screenshot_result = self.retry_with_cancellation(lambda: self.snapshot_and_save(folder_name, "03_result.png"), max_retries, retry_delay, "Step 12: Snapshot result")
            if screenshot_result is None:
                return False
            logger.info("✓ Step 12: Successfully captured result snapshot")

            # Step 13: ROI scan & compare (Post-battle with retry)
            logger.info("Step 13: ROI scan & compare (Post-battle)")
            post_battle_rois = FESTIVAL_CONFIG.get('post_battle_rois', ['獲得ザックマネー', '獲得アイテム', '獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース'])
            retry_count = 0
            is_ok_after = False
            while retry_count < max_retries:
                self.check_cancelled("Step 13: Post-battle verification")
                if use_detector and self.detector is not None:
                    extracted = self.scan_rois_combined(screenshot_result, post_battle_rois)
                    is_ok_after, msg, _ = self.compare_results(extracted, stage_data)
                    logger.info(f"Post-battle check (with detector): {msg}")
                else:
                    extracted = self.scan_screen_roi(screenshot_result, post_battle_rois)
                    is_ok_after, msg, _ = self.compare_results(extracted, stage_data, return_details=False)
                    logger.info(f"Post-battle check: {msg}")
                if is_ok_after:
                    logger.info("✓ Step 13: Post-battle verification passed")
                    break
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Step 13: Verification attempt {retry_count}/{max_retries} failed, retaking screenshot...")
                    screenshot_result = self.snapshot_and_save(folder_name, f"03_result_retry{retry_count}.png")
                    sleep(retry_delay)
            else:
                logger.error(f"Step 13: Post-battle verification failed after {max_retries} attempts")
                return False

            # Step 14: Touch OK to close result (first OK)
            logger.info("Step 14: Touch OK (close result - first)")
            self.touch_template("tpl_ok.png", optional=True)
            sleep(0.3)

            # Step 15: Touch OK to close result (second OK if needed)
            logger.info("Step 15: Touch OK (close result - second)")
            self.touch_template("tpl_ok.png", optional=True)
            sleep(0.3)

            final = is_ok_before and is_ok_after
            logger.info(f"{'='*60}\n{'✓ OK' if final else '✗ NG'}: Stage {stage_idx}\n{'='*60}")
            return final

        except CancellationError:
            logger.info(f"Stage {stage_idx} cancelled")
            return False
        except Exception as e:
            logger.error(f"✗ Stage {stage_idx}: {e}")
            return False

    def run_all_stages(self, data_path: str, output_path: Optional[str] = None,
                      use_detector: bool = False, resume: bool = True) -> bool:
        """
        Run automation for all stages with incremental saving and resume support.

        Args:
            data_path: Path to CSV/JSON file with test data
            output_path: Output result path (None = auto-generate)
            use_detector: Use detector (YOLO/Template)
            resume: Resume from existing results if available (default: True)

        Returns:
            bool: True if successful
        """
        # Initialize result_writer early to ensure it's available for error handling
        result_writer = None
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

            # Initialize ResultWriter with auto-write and resume support
            result_writer = ResultWriter(output_path, auto_write=True, resume=resume)

            # Log mode
            mode = "Detector + OCR" if use_detector and self.detector else "OCR only"
            logger.info(f"Mode: {mode} | Stages: {len(stages_data)} | Output: {output_path}")
            
            if resume and result_writer.completed_test_ids:
                logger.info(f"Resuming: {len(result_writer.completed_test_ids)} stages already completed")

            # Process each stage
            for idx, stage_data in enumerate(stages_data, 1):
                try:
                    self.check_cancelled(f"stage {idx}")
                except CancellationError:
                    logger.info(f"Cancellation requested, stopping at stage {idx}")
                    # Ensure results are saved before exiting
                    result_writer.flush()
                    result_writer.print_summary()
                    return False
                    
                test_case = stage_data.copy()
                test_case['test_case_id'] = idx

                # Skip if already completed (resume support)
                if resume and result_writer.is_completed(test_case):
                    logger.info(f"Stage {idx} already completed, skipping...")
                    continue

                is_ok = self.run_festival_stage(stage_data, idx, use_detector=use_detector)
                result_writer.add_result(test_case,
                                       ResultWriter.RESULT_OK if is_ok else ResultWriter.RESULT_NG,
                                       error_message=None if is_ok else "Verification failed")
                # Results are auto-saved after each add_result
                sleep(1.0)

            # Final save and summary
            result_writer.flush()
            result_writer.print_summary()
            return True

        except CancellationError:
            # Ensure results are saved on cancellation
            if result_writer:
                result_writer.flush()
                result_writer.print_summary()
            return False
        except Exception as e:
            logger.error(f"✗ All stages: {e}")
            # Ensure results are saved even on error
            if result_writer:
                result_writer.flush()
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

        try:
            self.check_cancelled("before starting")
        except CancellationError:
            return False

        if not self.agent.is_device_connected():
            logger.error("✗ Device not connected")
            return False

        try:
            success = self.run_all_stages(data_path, use_detector=use_detector)
        except CancellationError:
            logger.info("Automation cancelled")
            return False
        logger.info("="*70 + f"\n{'✓ COMPLETED' if success else '✗ FAILED'}\n" + "="*70)
        return success
