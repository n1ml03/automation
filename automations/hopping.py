"""
Hopping Automation

Flow cho Hopping (World Hopping):
1. touch(Template("tpl_world_map.png"))
2. touch(Template("tpl_hop_button.png"))
3. snapshot -> lưu folder (hop_01_before_hop.png)
4. touch(Template("tpl_confirm_hop.png"))
5. Wait for loading
6. snapshot -> lưu folder (hop_02_after_hop.png)
7. ROI scan -> kiểm tra world mới
8. Verify hop thành công
9. Lặp lại theo số lượng hops
"""

import os
import time
import cv2
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from airtest.core.api import Template
from core.agent import Agent
from core.utils import get_logger, ensure_directory
from core.data import ResultWriter, load_json, load_csv
from core.config import (
    HOPPING_ROI_CONFIG, get_hopping_roi_config,
    HOPPING_CONFIG, get_hopping_config, merge_config
)

logger = get_logger(__name__)


class HoppingAutomation:
    """Tự động hóa World Hopping."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        self.agent = agent
        
        # Merge config: base config từ HOPPING_CONFIG + custom config
        base_config = get_hopping_config()
        cfg = merge_config(base_config, config) if config else base_config

        # Paths
        self.templates_path = cfg.get('templates_path')
        self.snapshot_dir = cfg.get('snapshot_dir')
        self.results_dir = cfg.get('results_dir')
        
        # Timing
        self.wait_after_touch = cfg.get('wait_after_touch')
        self.loading_wait = cfg.get('loading_wait')
        self.cooldown_wait = cfg.get('cooldown_wait', 3.0)
        
        # Hop settings
        self.max_hops = cfg.get('max_hops', 10)
        self.retry_on_fail = cfg.get('retry_on_fail', True)
        self.max_retries = cfg.get('max_retries', 3)

        # Ensure directories exist
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)

        logger.info("HoppingAutomation initialized")

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

    def ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> str:
        """
        OCR một vùng ROI cụ thể cho Hopping.

        Args:
            roi_name: Tên ROI trong HOPPING_ROI_CONFIG
            screenshot: Screenshot để OCR, None = chụp mới

        Returns:
            str: Text được OCR từ vùng ROI (đã clean)
        """
        try:
            # Lấy config ROI
            roi_config = get_hopping_roi_config(roi_name)
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

    def scan_world_info(self, screenshot: Optional[Any] = None,
                       roi_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scan thông tin world hiện tại.

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
                roi_names = list(HOPPING_ROI_CONFIG.keys())

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

            logger.info(f"Scanned {len(results)} World ROIs")
            return results

        except Exception as e:
            logger.error(f"✗ Scan world info: {e}")
            return {}

    def verify_hop_success(self, before_world: str, after_world: str) -> bool:
        """Kiểm tra hop có thành công không (world có thay đổi)."""
        if not before_world or not after_world:
            return False

        # Nếu world name khác nhau thì hop thành công
        return before_world.strip().lower() != after_world.strip().lower()

    def run_world_hop(self, hop_data: Dict[str, Any], hop_idx: int) -> Dict[str, Any]:
        """Chạy một world hop."""
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
                world_info_before = self.scan_world_info(screenshot_before, ['world_name'])
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
            time.sleep(self.loading_wait)

            # Step 6: Check new world (after hop)
            logger.info("Step 6: Check new world")
            screenshot_after = self.snapshot_and_save(folder_name, f"{hop_idx:02d}_after.png")
            if screenshot_after is None:
                return result

            world_info_after = self.scan_world_info(screenshot_after, ['world_name'])
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

    def run_multiple_hops(self, num_hops: int, output_path: Optional[str] = None) -> bool:
        """Chạy nhiều world hops."""
        try:
            # Setup output
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.results_dir, f"hopping_results_{timestamp}.csv")

            result_writer = ResultWriter(output_path)
            logger.info(f"World hops: {num_hops} | Output: {output_path}")

            all_results = []
            successful_hops = 0

            # Process each hop
            for idx in range(1, num_hops + 1):
                hop_result = self.run_world_hop({}, idx)
                all_results.append(hop_result)

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

                time.sleep(2.0)  # Extra wait between hops

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

        except Exception as e:
            logger.error(f"✗ Multiple hops: {e}")
            return False

    def run(self, config: Dict[str, Any]) -> bool:
        """Entry point chính cho Hopping automation."""
        logger.info("="*60 + "\nHOPPING AUTOMATION START\n" + "="*60)

        if not self.agent.is_device_connected():
            logger.error("✗ Device not connected")
            return False

        num_hops = config.get('num_hops', 1)

        success = self.run_multiple_hops(num_hops)
        logger.info("="*60 + f"\n{'✓ COMPLETED' if success else '✗ FAILED'}\n" + "="*60)
        return success


if __name__ == '__main__':
    from core.agent import Agent

    agent = Agent()
    config = {
        'templates_path': './templates',
        'snapshot_dir': './result/hopping/snapshots',
        'results_dir': './result/hopping/results',
        'wait_after_touch': 1.0,
        'loading_wait': 5.0,
    }

    hopping = HoppingAutomation(agent, config)
    hopping_config = {
        'num_hops': 5
    }
    hopping.run(hopping_config)
