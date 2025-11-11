"""
Hopping Automation

Hopping flow (World Hopping):
1. touch(Template("tpl_world_map.png"))
2. touch(Template("tpl_hop_button.png"))
3. snapshot -> save to folder (hop_01_before_hop.png)
4. touch(Template("tpl_confirm_hop.png"))
5. Wait for loading
6. snapshot -> save to folder (hop_02_after_hop.png)
7. ROI scan -> check new world
8. Verify hop success
9. Repeat according to hop count
"""

import os
import time
import cv2
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from airtest.core.api import Template, exists, sleep
from core.agent import Agent
from core.utils import get_logger, ensure_directory
from core.data import ResultWriter, load_json, load_csv
from core.config import (
    HOPPING_ROI_CONFIG, get_hopping_roi_config,
    HOPPING_CONFIG, get_hopping_config, merge_config
)

logger = get_logger(__name__)


class HoppingAutomation:
    """Automate World Hopping."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        self.agent = agent
        
        # Merge config: base config from HOPPING_CONFIG + custom config
        base_config = get_hopping_config()
        cfg = merge_config(base_config, config) if config else base_config

        # Paths
        self.templates_path = cfg['templates_path']
        self.snapshot_dir = cfg['snapshot_dir']
        self.results_dir = cfg['results_dir']
        
        # Timing
        self.wait_after_touch = cfg['wait_after_touch']
        self.loading_wait = cfg['loading_wait']
        self.cooldown_wait = cfg['cooldown_wait']
        
        # Hop settings
        self.max_hops = cfg['max_hops']
        self.retry_on_fail = cfg['retry_on_fail']
        self.max_retries = cfg['max_retries']

        # Ensure directories exist
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)

        logger.info("HoppingAutomation initialized")

    def touch_template(self, template_name: str, optional: bool = False) -> bool:
        """Touch template image."""
        try:
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path):
                return optional

            if self.agent.device is None:
                logger.error("Device not connected")
                return False

            template = Template(template_path)
            pos = exists(template)

            if pos:
                self.agent.safe_touch(pos)
                logger.info(f"✓ {template_name}")
                sleep(self.wait_after_touch)
                return True

            return optional if optional else False

        except Exception as e:
            logger.error(f"✗ {template_name}: {e}") if not optional else None
            return optional

    def snapshot_and_save(self, folder_name: str, filename: str) -> Optional[Any]:
        """Take screenshot and save to folder."""
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

    def ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> str:
        """
        OCR specific ROI region for Hopping.

        Args:
            roi_name: ROI name in HOPPING_ROI_CONFIG
            screenshot: Screenshot for OCR, None = take new

        Returns:
            str: OCR text from ROI region (cleaned)
        """
        try:
            # Get ROI config
            roi_config = get_hopping_roi_config(roi_name)
            coords = roi_config['coords']  # [x1, x2, y1, y2]

            # Convert sang format (x1, y1, x2, y2) cho snapshot_region
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # Take screenshot or crop ROI region
            if screenshot is None:
                roi_image = self.agent.snapshot_region(region)
            else:
                # Crop from existing screenshot
                roi_image = screenshot[y1:y2, x1:x2]

            if roi_image is None:
                logger.warning(f"✗ ROI '{roi_name}': Cannot get image")
                return ""

            # OCR ROI region
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
        Clean OCR text (remove special chars, normalize).

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove newlines
        text = text.replace('\n', ' ').replace('\r', '')

        return text.strip()

    def scan_screen_roi(self, screenshot: Optional[Any] = None,
                       roi_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scan screen according to defined ROI regions (consistent with festivals).

        Args:
            screenshot: Screenshot to scan, None = take new
            roi_names: List of ROI names to scan, None = scan all

        Returns:
            Dict[str, Any]: Dictionary with ROI name as key, OCR text as value
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
                roi_names = list(HOPPING_ROI_CONFIG.keys())

            # OCR each ROI
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

    def verify_hop_success(self, before_world: str, after_world: str) -> bool:
        """Check if hop was successful (world changed)."""
        if not before_world or not after_world:
            return False

        # Hop successful if world names differ
        return before_world.strip().lower() != after_world.strip().lower()

    def run_hopping_stage(self, hop_data: Dict[str, Any], hop_idx: int) -> Dict[str, Any]:
        """Run hopping stage (consistent with festivals: run_xxx_stage)."""
        logger.info(f"\n{'='*50}\nHOP {hop_idx}\n{'='*50}")

        folder_name = f"hopping_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = {
            'hop_idx': hop_idx,
            'world_before': 'Unknown',
            'world_after': 'Unknown',
            'success': False
        }

        try:
            # Step 1: Check current world (before hop)
            logger.info("Step 1: Check current world")
            screenshot_before = self.snapshot_and_save(folder_name, f"{hop_idx:02d}_before.png")
            if screenshot_before is not None:
                world_info_before = self.scan_screen_roi(screenshot_before, ['world_name'])
                result['world_before'] = world_info_before.get('world_name', 'Unknown')

            # Step 2: Touch World Map
            logger.info("Step 2: Touch World Map")
            if not self.touch_template("tpl_world_map.png"):
                return result

            # Step 3: Touch Hop Button
            logger.info("Step 3: Touch Hop Button")
            if not self.touch_template("tpl_hop_button.png"):
                return result

            # Step 4: Confirm hop
            logger.info("Step 4: Confirm hop")
            if not self.touch_template("tpl_confirm_hop.png"):
                return result

            # Step 5: Wait for loading
            logger.info(f"Step 5: Wait for loading ({self.loading_wait}s)")
            sleep(self.loading_wait)

            # Step 6: Check new world (after hop)
            logger.info("Step 6: Check new world")
            screenshot_after = self.snapshot_and_save(folder_name, f"{hop_idx:02d}_after.png")
            if screenshot_after is None:
                return result

            world_info_after = self.scan_screen_roi(screenshot_after, ['world_name'])
            result['world_after'] = world_info_after.get('world_name', 'Unknown')

            # Step 7: Verify hop success
            logger.info("Step 7: Verify hop success")
            success = self.verify_hop_success(result['world_before'], result['world_after'])
            result['success'] = success

            if success:
                logger.info(f"✓ Hop successful: {result['world_before']} → {result['world_after']}")
            else:
                logger.warning(f"✗ Hop may have failed: {result['world_before']} → {result['world_after']}")

            logger.info(f"{'='*50}\n{'✓ SUCCESS' if success else '✗ FAILED'}: Hop {hop_idx}\n{'='*50}")
            return result

        except Exception as e:
            logger.error(f"✗ Hop {hop_idx}: {e}")
            return result

    def run_all_hops(self, data_path: Optional[str] = None, num_hops: Optional[int] = None,
                     output_path: Optional[str] = None) -> bool:
        """
        Run all hops (consistent with festivals: run_all_xxx).

        Args:
            data_path: Path to CSV/JSON file with test data (mode 1)
            num_hops: Number of hops to run (mode 2, if no data_path)
            output_path: Output result path (None = auto-generate)

        Returns:
            bool: True if successful
        """
        try:
            # Mode 1: Load from data file
            if data_path:
                test_data = load_json(data_path) if data_path.endswith('.json') else load_csv(data_path)
                if not test_data:
                    logger.error("✗ No data loaded")
                    return False
                
                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(self.results_dir, f"hopping_batch_{timestamp}.csv")
                
                result_writer = ResultWriter(output_path)
                logger.info(f"Batch sessions: {len(test_data)} | Output: {output_path}")
                
                all_success = True
                
                # Process each session from data
                for idx, session_data in enumerate(test_data, 1):
                    session_id = session_data.get('session_id', idx)
                    session_num_hops = int(session_data.get('num_hops', 1))
                    
                    # Override timing configs if provided
                    if 'loading_wait' in session_data:
                        self.loading_wait = float(session_data['loading_wait'])
                    if 'cooldown_wait' in session_data:
                        self.cooldown_wait = float(session_data['cooldown_wait'])
                    
                    logger.info(f"\n{'='*60}\nSESSION {idx}/{len(test_data)}: {session_num_hops} hops\n{'='*60}")
                    
                    # Run hops for this session
                    session_start = datetime.now()
                    successful_hops = 0
                    
                    for hop_idx in range(1, session_num_hops + 1):
                        hop_result = self.run_hopping_stage({}, hop_idx)
                        if hop_result['success']:
                            successful_hops += 1
                        sleep(2.0)
                    
                    session_end = datetime.now()
                    session_duration = (session_end - session_start).total_seconds()
                    session_success = successful_hops == session_num_hops
                    
                    if not session_success:
                        all_success = False
                    
                    # Add session summary
                    result_writer.add_result(
                        test_case={
                            'session_id': session_id,
                            'num_hops': session_num_hops,
                            'successful_hops': successful_hops,
                            'failed_hops': session_num_hops - successful_hops,
                            'success_rate': f"{(successful_hops/session_num_hops*100):.1f}%",
                            'duration_seconds': f"{session_duration:.1f}",
                        },
                        result=ResultWriter.RESULT_OK if session_success else ResultWriter.RESULT_NG,
                        error_message=None if session_success else f"Only {successful_hops}/{session_num_hops} hops succeeded"
                    )
                    
                    logger.info(f"Session {idx} completed: {successful_hops}/{session_num_hops} successful")
                    sleep(3.0)
                
                # Save results
                result_writer.write()
                result_writer.print_summary()
                return all_success
            
            # Mode 2: Direct num_hops
            elif num_hops:
                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(self.results_dir, f"hopping_results_{timestamp}.csv")

                result_writer = ResultWriter(output_path)
                logger.info(f"World hops: {num_hops} | Output: {output_path}")

                successful_hops = 0

                # Process each hop
                for idx in range(1, num_hops + 1):
                    hop_result = self.run_hopping_stage({}, idx)

                    if hop_result['success']:
                        successful_hops += 1

                    # Add to result writer
                    test_case = {
                        'hop_number': idx,
                        'world_before': hop_result.get('world_before', ''),
                        'world_after': hop_result.get('world_after', ''),
                        'success': hop_result.get('success', False)
                    }
                    result_writer.add_result(test_case,
                                           ResultWriter.RESULT_OK if hop_result['success'] else ResultWriter.RESULT_NG,
                                           error_message=None if hop_result['success'] else "Hop verification failed")

                    sleep(2.0)

                # Save results
                result_writer.write()
                result_writer.print_summary()

                # Summary statistics
                success_rate = (successful_hops / num_hops) * 100 if num_hops > 0 else 0
                logger.info(f"\nHopping Summary:")
                logger.info(f"Total hops: {num_hops}")
                logger.info(f"Successful: {successful_hops}")
                logger.info(f"Success rate: {success_rate:.1f}%")

                return True
            
            else:
                logger.error("✗ Either 'data_path' or 'num_hops' must be provided")
                return False

        except Exception as e:
            logger.error(f"✗ Run all hops: {e}")
            return False

    def run(self, config: Optional[Dict[str, Any]] = None, data_path: Optional[str] = None) -> bool:
        """
        Main entry point for Hopping automation (consistent with festivals).

        Supports 2 modes:
        1. Config mode: Pass config dict with num_hops
        2. Data mode: Pass data_path to CSV/JSON file

        Args:
            config: Configuration dict (mode 1)
            data_path: Path to CSV/JSON data file (mode 2)

        Returns:
            bool: True if successful
            
        Example usage:
            # Mode 1: Direct config
            hopping.run(config={'num_hops': 5})
            
            # Mode 2: Load from file
            hopping.run(data_path='./data/hopping_tests.csv')
        """
        logger.info("="*60 + "\nHOPPING AUTOMATION START\n" + "="*60)

        if not self.agent.is_device_connected():
            logger.error("✗ Device not connected")
            return False
        
        # Mode 2: Load from data file
        if data_path:
            success = self.run_all_hops(data_path=data_path)
        # Mode 1: Direct config
        elif config:
            num_hops = config.get('num_hops', 1)
            success = self.run_all_hops(num_hops=num_hops)
        else:
            logger.error("✗ Either 'config' or 'data_path' must be provided")
            return False
        
        logger.info("="*60 + f"\n{'✓ COMPLETED' if success else '✗ FAILED'}\n" + "="*60)
        return success